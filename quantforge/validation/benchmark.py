from pathlib import Path

import pandas as pd


class BenchmarkEngine:

    def __init__(self, prediction_file, holding_days):

        self.df = pd.read_csv(
            prediction_file,
            low_memory=False,
        )

        self.df["Date"] = pd.to_datetime(
            self.df["Date"]
        )

        self.holding_days = holding_days

    def equal_weight(self):
        return (
            self.df
            .groupby("Date")["RETURN_1D"]
            .mean()
        )

    def top_n(
        self,
        n=15,
    ):

        x = (
            self.df
            .sort_values(
                ["Date", "PRED_RETURN"],
                ascending=[True, False],
            )
            .groupby("Date")
            .head(n)
        )

        return (
            x.groupby("Date")["RETURN_1D"]
            .mean()
        )

    def buy_and_hold(self):
        # Compute market return as equal-weighted average of all stocks' daily returns
        return (
            self.df
            .groupby("Date")["RETURN_1D"]
            .mean()
        )

    def equity(
        self,
        returns,
        holding_days,
    ):

        returns = returns.iloc[::holding_days]

        return (
            1.0 + returns.fillna(0)
        ).cumprod()

    def report(self):

        ew = self.equity(
            self.equal_weight(),
            self.holding_days,
        )

        top = self.equity(
            self.top_n(),
            self.holding_days,
        )

        market = self.equity(
            self.buy_and_hold(),
            self.holding_days,
        )

        out = pd.DataFrame(
            {
                "EqualWeight": ew,
                "TopN": top,
                "Market": market,
            }
        )

        Path("results").mkdir(
            exist_ok=True
        )

        out.to_csv(
            "results/benchmark.csv"
        )

        print()
        print("=" * 80)
        print("BENCHMARK")
        print("=" * 80)
        print(out.tail())
        print()
