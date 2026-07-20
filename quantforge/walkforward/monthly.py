from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from quantforge.training.model_manager import ModelManager
from quantforge.walkforward.checkpoint import CheckpointManager


def _train_fold_worker(fold_data):
    date = fold_data["date"]
    train_idx = fold_data["train_idx"]
    test_idx = fold_data["test_idx"]
    feature_matrix = fold_data["feature_matrix"]
    target_vector = fold_data["target_vector"]
    model_config = fold_data["model_config"]
    features = fold_data["features"]
    target = fold_data["target"]
    test_df = fold_data["test_df"]

    X_train = feature_matrix[train_idx]
    y_train = target_vector[train_idx]
    X_test = feature_matrix[test_idx]

    model_manager = ModelManager(model_config)
    model = model_manager.build()
    model.fit(X_train, y_train)

    pred_series = model.predict(X_test)

    test_pred = test_df[['Date', 'Ticker', target, 'RETURN_1D', 'VOL_20D']].copy()
    test_pred['PRED_RETURN'] = pred_series

    return {
        "date": date,
        "predictions": test_pred,
    }


class MonthlyLoop:
    def __init__(
        self,
        df,
        features,
        target,
        model_manager,
        checkpoint_manager,
        prediction_file,
        purge_days=5,
        train_length=252,
        test_length=20,
        step=20,
        workers=8,
        dashboard=None,
    ):
        self.df = df.copy()
        self.df = self.df.sort_values(["Date", "Ticker"], kind="mergesort").reset_index(drop=True)

        self.features = features
        self.target = target
        self.model_manager = model_manager
        self.checkpoint_manager = checkpoint_manager
        self.prediction_file = Path(prediction_file)
        self.purge_days = purge_days
        self.train_length = train_length
        self.test_length = test_length
        self.step = step
        self.workers = workers
        self.dashboard = dashboard

        self.prediction_file.parent.mkdir(parents=True, exist_ok=True)

        self.dates = [
            pd.Timestamp(x)
            for x in self.df["Date"].drop_duplicates().sort_values().to_numpy()
        ]

        self.feature_matrix = self.df[self.features].to_numpy(dtype=np.float32, copy=False)
        self.target_vector = self.df[self.target].to_numpy(dtype=np.float32, copy=False)

        # Keep only necessary columns for the test_df
        self.df_for_test = self.df[['Date', 'Ticker', self.target, 'RETURN_1D', 'VOL_20D']].copy()

        self.model_config = {
            k: v for k, v in model_manager.config.items()
            if k not in ["data_path", "prediction_file", "checkpoint_file"]
        }

        self.save_queue = queue.Queue()
        self.writer_thread = None

    def _split_date(self, date):
        date = pd.Timestamp(date)
        train_start = date - timedelta(days=self.train_length)
        train_end = date - timedelta(days=1)
        test_start = date
        test_end = date + timedelta(days=self.test_length - 1)
        return train_start, train_end, test_start, test_end

    def _save_predictions(self, date, predictions):
        predictions = predictions.copy()
        predictions["Date"] = date
        out_dir = self.prediction_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"pred_{pd.Timestamp(date).strftime('%Y%m%d')}.parquet"
        predictions.to_parquet(fname, index=False, compression="zstd")

    def _merge_predictions(self):
        files = sorted(self.prediction_file.parent.glob("pred_*.parquet"))
        if not files:
            return
        merged = None
        for f in files:
            df = pd.read_parquet(f)
            if merged is None:
                merged = df
            else:
                merged = pd.concat([merged, df], ignore_index=True)
        if merged is not None:
            merged.to_parquet(self.prediction_file, index=False, compression="zstd")

    def _writer_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            date, predictions = item
            self._save_predictions(date, predictions)

    def run(self):
        if self.dashboard:
            self.dashboard.start_timer("walkforward")

        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

        completed = self.checkpoint_manager.get_completed_dates()
        if completed:
            print("=" * 80)
            print(f"Recovered {len(completed)} completed checkpoints")
            print("=" * 80)

        start_idx = self.train_length + self.test_length

        folds = []
        for i in range(start_idx, len(self.dates), self.step):
            date = self.dates[i]
            date_ts = pd.Timestamp(date)
            if date_ts in completed:
                print(f"Skipping {date_ts.date()} (already completed)")
                continue

            train_start, train_end, test_start, test_end = self._split_date(date)

            train_mask = (
                (self.df["Date"] >= train_start) &
                (self.df["Date"] <= train_end)
            )
            test_mask = (
                (self.df["Date"] >= test_start) &
                (self.df["Date"] <= test_end)
            )

            train_idx = np.where(train_mask.to_numpy())[0]
            test_idx = np.where(test_mask.to_numpy())[0]

            # ---- Filter test rows with NaN features ----
            feature_block = self.feature_matrix[test_idx]
            valid_mask = ~np.isnan(feature_block).any(axis=1)
            test_idx = test_idx[valid_mask]

            if len(test_idx) == 0:
                continue

            test_df = self.df_for_test.iloc[test_idx].copy()
            test_indices = test_idx

            folds.append({
                "date": date,
                "train_idx": train_idx,
                "test_idx": test_indices,
                "feature_matrix": self.feature_matrix,
                "target_vector": self.target_vector,
                "model_config": self.model_config,
                "features": self.features,
                "target": self.target,
                "test_df": test_df,
            })

        if not folds:
            print("No new months to process.")
            self.save_queue.put(None)
            self.writer_thread.join()
            if self.dashboard:
                self.dashboard.stop_timer("walkforward")
                self.dashboard.record("folds_processed", 0)
            return self.checkpoint_manager

        n_folds = len(folds)
        workers = min(self.workers, n_folds, os.cpu_count() or 1)
        print(f"Processing {n_folds} months with {workers} workers...")

        fold_times = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_train_fold_worker, fold): fold["date"] for fold in folds}

            for future in as_completed(futures):
                t0 = time.perf_counter()
                result = future.result()
                elapsed = time.perf_counter() - t0
                fold_times.append(elapsed)

                date = result["date"]
                preds = result["predictions"]
                self.save_queue.put((date, preds))

        if fold_times:
            avg_fold = sum(fold_times) / len(fold_times)
            total_fit = sum(fold_times)
            if self.dashboard:
                self.dashboard.record("avg_fold_seconds", avg_fold)
                self.dashboard.record("total_fit_seconds", total_fit)

        # Signal writer to stop and wait
        self.save_queue.put(None)
        self.writer_thread.join()

        # Now merge all predictions
        merge_start = time.perf_counter()
        self._merge_predictions()
        merge_time = time.perf_counter() - merge_start
        if self.dashboard:
            self.dashboard.record("merge_seconds", merge_time)

        self.checkpoint_manager.finalize()

        if self.dashboard:
            self.dashboard.stop_timer("walkforward")
            self.dashboard.record("folds_processed", n_folds)

        print("Walk-forward monthly loop completed.")
        return self.checkpoint_manager
