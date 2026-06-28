from pathlib import Path

import pandas as pd

from quantforge.portfolio.equal_weight import (
    build_equal_weight_portfolio,
)

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

        portfolio = build_equal_weight_portfolio(

            predictions,

            score_column=self.config.get(
                "score_column",
                "PRED_RETURN",
            ),

            top_n=self.config["top_n"],

        )

        metrics = evaluate(

            portfolio,

            holding_days=self.config[
                "holding_days"
            ],

        )

        return portfolio, metrics