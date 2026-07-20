from pathlib import Path
import json

import pandas as pd


class ExperimentManager:

    def __init__(self, config_path):

        self.config_path = Path(config_path)

        with open(self.config_path) as f:
            self.config = json.load(f)

        self.predictions = None

    def load_predictions(self):

        if self.predictions is None:

            path = Path(
                self.config["prediction_file"]
            )

            self.predictions = pd.read_csv(path, low_memory=False)

            self.predictions["Date"] = pd.to_datetime(
                self.predictions["Date"]
            )

            print(
                f"Loaded {len(self.predictions):,} rows once."
            )

        return self.predictions.copy()
