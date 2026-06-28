from pathlib import Path
import pandas as pd


class DatasetBuilder:
    """
    Loads and prepares the training dataset.

    Responsibilities
    ----------------
    - Load CSV
    - Parse dates
    - Sort rows
    - Drop NA
    - Optimize memory
    """

    def __init__(
        self,
        data_path,
        features,
        target,
    ):

        self.data_path = Path(data_path)
        self.features = features
        self.target = target

    def load(self):

        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)

        df = pd.read_csv(self.data_path)

        df["Date"] = pd.to_datetime(df["Date"])

        df = (
            df
            .sort_values(
                ["Date", "Ticker"]
            )
            .reset_index(drop=True)
        )

        return df

    def prepare(self):

        df = self.load()

        df = df.dropna(
            subset=self.features + [self.target]
        )

        for c in self.features:

            df[c] = df[c].astype("float32")

        df[self.target] = (
            df[self.target]
            .astype("float32")
        )

        print()

        print("Rows :", len(df))
        print("Cols :", len(df.columns))
        print("Features :", len(self.features))

        return df