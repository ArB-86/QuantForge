from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import time
import os
import multiprocessing as mp
import joblib

from quantforge.training.model_manager import ModelManager
from quantforge.walkforward.checkpoint import CheckpointManager

# Global arrays – inherited read‑only via fork
_G_FEAT = None
_G_TARG = None
_G_DF   = None

def _worker(gpu_id: int, chunk: list, out_dir: str):
    """Process a chunk of folds on a fixed GPU – writes predictions + model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    feature_matrix = _G_FEAT
    target_vector  = _G_TARG
    df_for_test    = _G_DF
    print(f"Shapes: {feature_matrix.shape} {target_vector.shape} {df_for_test.shape}")

    # Temporary directory for models (one per fold)
    tmp_model_dir = Path(out_dir) / "tmp_models"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    for task in chunk:
        date, train_idx, test_idx, model_config, target = task

        print(f"[GPU {gpu_id}] fitting {len(train_idx)} rows – {date}")

        X_train = np.take(feature_matrix, train_idx, axis=0)
        y_train = np.take(target_vector, train_idx, axis=0)
        X_test  = np.take(feature_matrix, test_idx, axis=0)
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_test  = np.ascontiguousarray(X_test, dtype=np.float32)
        y_train = np.ascontiguousarray(y_train, dtype=np.float32)

        test_df = df_for_test.iloc[test_idx].copy()

        mm = ModelManager(model_config)
        model = mm.build()
        model.fit(X_train, y_train)

        # Save model immediately (per fold, to temp location)
        model_filename = tmp_model_dir / f"model_{pd.Timestamp(date).strftime('%Y%m%d')}.pkl"
        joblib.dump(model, model_filename)

        print(f"[GPU {gpu_id}] {date} finished – model saved")

        pred = model.predict(X_test)
        pred_df = test_df[['Date', 'Ticker', target, 'RETURN_1D', 'VOL_20D']].copy()
        pred_df['PRED_RETURN'] = pred

        fname = Path(out_dir) / f"pred_{pd.Timestamp(date).strftime('%Y%m%d')}.parquet"
        pred_df.to_parquet(fname, index=False, compression="zstd")

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

        # Store model file path from config (for final model dump)
        self.model_file = model_manager.config.get("model_file")
        if self.model_file:
            self.model_file = Path(self.model_file)
            self.model_file.parent.mkdir(parents=True, exist_ok=True)

        self.dates = [
            pd.Timestamp(x)
            for x in self.df["Date"].drop_duplicates().sort_values().to_numpy()
        ]

        self.feature_matrix = self.df[self.features].to_numpy(dtype=np.float32, copy=False)
        self.target_vector  = self.df[self.target].to_numpy(dtype=np.float32, copy=False)
        self.df_for_test    = self.df[['Date', 'Ticker', self.target, 'RETURN_1D', 'VOL_20D']].copy()

        # Model config passed to workers (without file paths)
        self.model_config = {
            k: v for k, v in model_manager.config.items()
            if k not in ["data_path", "prediction_file", "checkpoint_file", "model_file"]
        }

    def _split_date(self, date):
        date = pd.Timestamp(date)
        train_start = date - timedelta(days=self.train_length)
        train_end   = date - timedelta(days=1)
        test_start  = date
        test_end    = date + timedelta(days=self.test_length - 1)
        return train_start, train_end, test_start, test_end

    def _merge_predictions(self):
        files = sorted(self.prediction_file.parent.glob("pred_*.parquet"))
        if not files:
            return
        frames = [pd.read_parquet(f) for f in files]
        merged = pd.concat(frames, ignore_index=True, copy=False)
        merged.to_parquet(self.prediction_file, index=False, compression="zstd")

    def run(self):
        overall = time.perf_counter()
        if self.dashboard:
            self.dashboard.start_timer("walkforward")

        completed = self.checkpoint_manager.get_completed_dates()
        if completed:
            print("=" * 80)
            print(f"Recovered {len(completed)} completed checkpoints")
            print("=" * 80)

        start_idx = self.train_length + self.test_length

        # Build task list (only indices + config)
        tasks = []
        for i in range(start_idx, len(self.dates), self.step):
            date = self.dates[i]
            date_ts = pd.Timestamp(date)
            if date_ts in completed:
                print(f"Skipping {date_ts.date()} (already completed)")
                continue

            train_start, train_end, test_start, test_end = self._split_date(date)
            train_mask = (self.df["Date"] >= train_start) & (self.df["Date"] <= train_end)
            test_mask  = (self.df["Date"] >= test_start)  & (self.df["Date"] <= test_end)

            train_idx = np.where(train_mask.to_numpy())[0]
            test_idx  = np.where(test_mask.to_numpy())[0]

            feature_block = self.feature_matrix[test_idx]
            valid_mask = ~np.isnan(feature_block).any(axis=1)
            test_idx = test_idx[valid_mask]
            if len(test_idx) == 0:
                continue

            tasks.append((
                date,
                train_idx,
                test_idx,
                self.model_config.copy(),
                self.target,
            ))

        fold_gen_time = time.perf_counter() - overall
        print(f"Fold generation : {fold_gen_time:.2f}s")
        n_folds = len(tasks)

        if n_folds == 0:
            print("No new months to process.")
            if self.dashboard:
                self.dashboard.stop_timer("walkforward")
                self.dashboard.record("folds_processed", 0)
            return self.checkpoint_manager

        print("Windows compatibility: running walkforward sequentially")

        global _G_FEAT, _G_TARG, _G_DF
        _G_FEAT = self.feature_matrix
        _G_TARG = self.target_vector
        _G_DF = self.df_for_test

        out_dir = str(self.prediction_file.parent)

        _worker(0, tasks, out_dir)
        train_time = time.perf_counter() - overall - fold_gen_time
        print(f"Training (parallel) : {train_time:.2f}s")

        # Merge predictions
        merge_start = time.perf_counter()
        self._merge_predictions()
        merge_time = time.perf_counter() - merge_start
        print(f"Merge : {merge_time:.2f}s")

        # Finalise model: pick the latest model by date from tmp_models and save as final model
        if self.model_file:
            tmp_model_dir = Path(out_dir) / "tmp_models"
            model_files = sorted(tmp_model_dir.glob("model_*.pkl"))
            if model_files:
                latest_model = model_files[-1]   # alphabetical by date works
                print(f"Publishing final model from {latest_model.name}")
                # Copy to final location
                joblib.dump(joblib.load(latest_model), self.model_file)
            else:
                print("No model files found – skip final model save.")

        self.checkpoint_manager.finalize()

        if self.dashboard:
            self.dashboard.stop_timer("walkforward")
            self.dashboard.record("folds_processed", n_folds)

        total_time = time.perf_counter() - overall
        print(f"TOTAL : {total_time:.2f}s")
        print("Walk-forward monthly loop completed.")
        return self.checkpoint_manager