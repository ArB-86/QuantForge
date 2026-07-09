from pathlib import Path

import joblib
import pandas as pd


class CheckpointManager:

    def __init__(
        self,
        checkpoint_file,
        model_file,
    ):

        self.checkpoint_file = Path(checkpoint_file)
        self.model_file = Path(model_file)

        self.checkpoint_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def completed_months(self):

        if not self.checkpoint_file.exists():

            return set()

        old = pd.read_csv(
            self.checkpoint_file,
            usecols=["Date"],
        )

        old["Date"] = pd.to_datetime(
            old["Date"]
        )

        return set(
            old["Date"]
            .dt
            .to_period("M")
            .astype(str)
            .unique()
        )

    def append_predictions(
        self,
        df,
    ):

        df.to_csv(
            self.checkpoint_file,
            mode="a",
            header=not self.checkpoint_file.exists(),
            index=False,
        )

    def save_model(
        self,
        model,
    ):

        joblib.dump(
            model,
            self.model_file,
        )

        print()

        print(
            "Model saved:",
            self.model_file,
        )
