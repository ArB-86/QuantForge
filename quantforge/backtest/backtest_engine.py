from pathlib import Path

import pandas as pd

from quantforge.portfolio.allocator import (
    build_portfolio,
)

from quantforge.backtest.simulator import simulate
from quantforge.backtest.metrics import evaluate


class BacktestEngine:

    def __init__(self, config):

        self.config = config

    def load_predictions(self):

        df = pd.read_csv(
            Path(
                self.config["prediction_file"]
            )
        )

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        return df

    def run(self):

        predictions = self.load_predictions()

        portfolio = build_portfolio(

            predictions,

            method=self.config.get(
                "portfolio",
                "equal_weight",
            ),

            score_column=self.config.get(
                "score_column",
                "PRED_RETURN",
            ),

            top_n=self.config["top_n"],

        )

        print("=" * 80)
        print("Portfolio Method:", self.config["portfolio"])
        print(portfolio[["Date", "Ticker", "Weight"]].head(20))
        print("=" * 80)

        portfolio = simulate(

            portfolio,

            return_column=self.config["target"],

            holding_days=self.config["holding_days"],

            round_trip_cost=self.config["transaction_cost"],

        )

        metrics = evaluate(

            portfolio,

            holding_days=self.config["holding_days"],

        )

        return portfolio, metrics