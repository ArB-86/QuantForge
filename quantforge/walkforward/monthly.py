from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os

from quantforge.training.model_manager import ModelManager
from quantforge.walkforward.checkpoint import CheckpointManager


def _train_fold_worker(fold_data):
    """
    Worker function: trains on a fold using shared NumPy arrays.
    Input: dict with train_idx, test_idx, date, feature_matrix, target_vector, model_config, features, target, test_df
    """
    date = fold_data["date"]
    train_idx = fold_data["train_idx"]
    test_idx = fold_data["test_idx"]
    feature_matrix = fold_data["feature_matrix"]
    target_vector = fold_data["target_vector"]
    model_config = fold_data["model_config"]
    features = fold_data["features"]
    target = fold_data["target"]
    test_df = fold_data["test_df"]  # contains Date, Ticker, target, etc.

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
    """
    Monthly walk-forward training loop with shared-memory optimizations.
    """

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
    ):
        # Store a flat copy without setting index
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

        self.prediction_file.parent.mkdir(parents=True, exist_ok=True)

        # Pre-compute unique dates
        self.dates = self.df["Date"].drop_duplicates().sort_values().to_numpy()

        # ---- Shared memory: convert to NumPy once ----
        self.feature_matrix = self.df[self.features].to_numpy(dtype=np.float32, copy=False)
        self.target_vector = self.df[self.target].to_numpy(dtype=np.float32, copy=False)

        # We'll also keep the full DataFrame for test_df extraction
        self.df_for_test = self.df[['Date', 'Ticker', self.target, 'RETURN_1D', 'VOL_20D']].copy()

        # Model config for workers (without large objects)
        self.model_config = {
            k: v for k, v in model_manager.config.items()
            if k not in ["data_path", "prediction_file", "checkpoint_file"]
        }

        # Background writer queue
        self.save_queue = queue.Queue()
        self.writer_thread = None

    def _split_date(self, date):
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
        # Use incremental merge to avoid loading all files at once
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
        """Background thread to save predictions."""
        while True:
            item = self.save_queue.get()
            if item is None:  # sentinel to stop
                break
            date, predictions = item
            self._save_predictions(date, predictions)

    def run(self):
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

        completed = self.checkpoint_manager.get_completed_dates()
        if completed:
            print("=" * 80)
            print(f"Recovered {len(completed)} completed checkpoints")
            print("=" * 80)

        start_idx = self.train_length + self.test_length

        # Prepare fold data as list of (date, train_mask, test_mask)
        folds = []
        for i in range(start_idx, len(self.dates), self.step):
            date = self.dates[i]
            date_ts = pd.Timestamp(date)
            if date_ts in completed:
                print(f"Skipping {date_ts.date()} (already completed)")
                continue

            train_start, train_end, test_start, test_end = self._split_date(date)

            # Boolean masks
            train_mask = (
                (self.df["Date"] >= train_start) &
                (self.df["Date"] <= train_end)
            )
            test_mask = (
                (self.df["Date"] >= test_start) &
                (self.df["Date"] <= test_end)
            )

            # Get indices
            train_idx = np.where(train_mask.to_numpy())[0]
            test_idx = np.where(test_mask.to_numpy())[0]

            # Drop NaN in test features: we need to filter test_idx further
            # Instead, we'll filter the test DataFrame directly.
            # For simplicity, we'll use the entire test_idx and let the worker handle it.
            # But we need to pass the test DataFrame.
            # We'll pass a small copy of test_df for the worker.
            test_df = self.df_for_test.iloc[test_idx].copy()
            test_df = test_df.dropna(subset=self.features)

            # Update indices after dropna
            # We need to realign with the original test_idx: we can just pass the filtered test_df.
            # For the worker, we'll send test_df as is (it's small).
            # And we'll send train_idx and test_idx (full) but then filter in worker.
            # Actually, the worker should use the filtered test_df's indices relative to the full feature_matrix.
            # Simpler: we'll pass the test_df directly and the worker uses it.

            # We'll pass test_df, and also pass train_idx and test_idx (before filtering)
            # The worker will filter test_idx based on test_df's rows? That's messy.
            # Better: in the worker, we'll use the test_df's index to slice feature_matrix.
            # We'll pass test_df's index, but feature_matrix is aligned with self.df index.
            # So we need to pass the original indices of test_df rows.
            test_indices = test_df.index.values
            train_indices = train_idx

            # Since test_df is a subset, we can pass its index.
            # And also the full feature_matrix.
            folds.append({
                "date": date,
                "train_idx": train_indices,
                "test_idx": test_indices,  # indices in the original df
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
            return self.checkpoint_manager

        # Dynamic worker count
        n_folds = len(folds)
        workers = min(self.workers, n_folds, os.cpu_count() or 1)
        print(f"Processing {n_folds} months with {workers} workers...")

        # Use ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_train_fold_worker, fold): fold["date"] for fold in folds}

            for future in as_completed(futures):
                result = future.result()
                date = result["date"]
                preds = result["predictions"]
                # Put into background writer queue
                self.save_queue.put((date, preds))

        # Signal writer to stop
        self.save_queue.put(None)
        self.writer_thread.join()

        # Merge predictions
        self._merge_predictions()

        # Finalize checkpoints
        self.checkpoint_manager.finalize()

        print("Walk-forward monthly loop completed.")
        return self.checkpoint_manager
