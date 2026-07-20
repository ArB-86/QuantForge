from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

from quantforge.training.model_manager import ModelManager
from quantforge.walkforward.checkpoint import CheckpointManager


def _run_single_fold(fold_data):
    """
    Worker function: train one fold and return predictions.
    This runs in a separate process.
    """
    # Unpack fold data
    date = fold_data["date"]
    train = fold_data["train"]
    test = fold_data["test"]
    features = fold_data["features"]
    target = fold_data["target"]
    model_config = fold_data["model_config"]

    # Train model
    X_train = train[features]
    y_train = train[target]

    model_manager = ModelManager(model_config)
    model = model_manager.build()
    model.fit(X_train, y_train)

    # Predict
    X_test = test[features]
    pred_series = model.predict(X_test)

    test_pred = test[['Date', 'Ticker', target, 'RETURN_1D', 'VOL_20D']].copy()
    test_pred['PRED_RETURN'] = pred_series

    return {
        "date": date,
        "predictions": test_pred,
        "model": model,
    }


class MonthlyLoop:
    """
    Monthly walk-forward training loop with parallel execution.
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
        # Sort and set index once
        self.df = df.copy()
        self.df = self.df.sort_values(["Date", "Ticker"], kind="mergesort")
        self.df = self.df.set_index("Date", drop=False)

        self.features = features
        self.target = target
        self.model_manager = model_manager
        self.checkpoint_manager = checkpoint_manager
        self.prediction_file = Path(prediction_file)
        self.purge_days = purge_days
        self.train_length = train_length
        self.test_length = test_length
        self.step = step
        self.workers = min(workers, os.cpu_count() or 1)

        self.prediction_file.parent.mkdir(parents=True, exist_ok=True)

        # Cache unique dates once
        self.dates = self.df["Date"].drop_duplicates().sort_values().to_numpy()

        # Model config for workers
        self.model_config = {
            k: v for k, v in model_manager.config.items()
            if k not in ["data_path", "prediction_file", "checkpoint_file"]
        }

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
        df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
        df.to_parquet(self.prediction_file, index=False, compression="zstd")

    def run(self):
        completed = self.checkpoint_manager.get_completed_dates()
        if completed:
            print("=" * 80)
            print(f"Recovered {len(completed)} completed checkpoints")
            print("=" * 80)

        start_idx = self.train_length + self.test_length

        # Prepare folds
        folds = []
        for i in range(start_idx, len(self.dates), self.step):
            date = self.dates[i]
            if pd.Timestamp(date) in completed:
                print(f"Skipping {pd.Timestamp(date).date()} (already completed)")
                continue

            train_start, train_end, test_start, test_end = self._split_date(date)

            # Use DateIndex slicing instead of boolean masks
            train = self.df.loc[train_start:train_end].copy()
            test = self.df.loc[test_start:test_end].copy()

            # Drop rows with NaN features in test set
            test = test.dropna(subset=self.features)

            if len(train) == 0 or len(test) == 0:
                continue

            folds.append({
                "date": date,
                "train": train,
                "test": test,
                "features": self.features,
                "target": self.target,
                "model_config": self.model_config,
            })

        if not folds:
            print("No new months to process.")
            return self.checkpoint_manager

        print(f"Processing {len(folds)} months with {self.workers} workers...")

        # Parallel execution
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(_run_single_fold, fold) for fold in folds]

            for future in futures:
                result = future.result()
                date = result["date"]
                predictions = result["predictions"]
                self._save_predictions(date, predictions)

        # Merge all per-date predictions
        self._merge_predictions()

        # Finalize checkpoint manager
        self.checkpoint_manager.finalize()

        print("Walk-forward monthly loop completed.")
        return self.checkpoint_manager
