import pandas as pd
from pathlib import Path
import joblib


class CheckpointManager:

    def __init__(self, checkpoint_file, model_file, flush_interval=10):
        self.checkpoint_file = Path(checkpoint_file)
        self.model_file = Path(model_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_file.parent.mkdir(parents=True, exist_ok=True)

        self._buffer = []
        self.flush_interval = flush_interval

    def _flush(self):
        if not self._buffer:
            return
        new_rows = pd.DataFrame(self._buffer)
        if self.checkpoint_file.exists():
            existing = pd.read_csv(self.checkpoint_file)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined.to_csv(self.checkpoint_file, index=False)
        else:
            new_rows.to_csv(self.checkpoint_file, index=False)
        self._buffer.clear()

    def save(self, model, date, train_dates, test_dates):
        # Save model (overwrite)
        joblib.dump(model, self.model_file)

        # Store checkpoint metadata in buffer
        train_start, train_end = train_dates
        test_start, test_end = test_dates
        self._buffer.append({
            'Date': date,
            'Train_Start': train_start,
            'Train_End': train_end,
            'Test_Start': test_start,
            'Test_End': test_end,
        })

        # Flush if buffer size exceeds interval
        if len(self._buffer) >= self.flush_interval:
            self._flush()

    def load(self):
        if self.model_file.exists():
            return joblib.load(self.model_file)
        return None

    def get_latest_checkpoint(self):
        if self.checkpoint_file.exists():
            df = pd.read_csv(self.checkpoint_file)
            return df['Date'].max()
        return None

    def get_completed_dates(self):
        if not self.checkpoint_file.exists():
            return set()
        df = pd.read_csv(self.checkpoint_file, parse_dates=['Date'])
        return {pd.Timestamp(x) for x in df['Date']}

    def finalize(self):
        """Call this at the end of the run to flush any remaining buffer."""
        self._flush()
def finalize(self): self._flush()
