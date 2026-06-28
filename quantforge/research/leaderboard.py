
import json
from pathlib import Path
import pandas as pd


class Leaderboard:

    def __init__(
        self,
        folder="results"
    ):

        self.folder = Path(folder)

    def load(self):

        rows = []

        for f in self.folder.glob("*.json"):

            with open(f) as fp:

                rows.append(
                    json.load(fp)
                )

        if len(rows) == 0:

            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values(
                "sharpe",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


if __name__ == "__main__":

    lb = Leaderboard()

    df = lb.load()

    if len(df):

        print()

        print(
            df[
                [
                    "name",
                    "model",
                    "sharpe",
                    "cagr",
                    "max_drawdown",
                    "win_rate"
                ]
            ]
        )

    else:

        print("No experiments found.")
