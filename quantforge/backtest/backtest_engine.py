from pathlib import Path

import pandas as pd

from quantforge.portfolio_engine.allocator import (
    build_portfolio,
)
from quantforge.portfolio_engine.constraints import PortfolioConstraints
from quantforge.portfolio_engine.volatility_target import (
    VolatilityTarget,
)

from quantforge.backtest.simulator import simulate
from quantforge.backtest.metrics import evaluate


class BacktestEngine:

    def __init__(self, config):

        self.config = config

    def load_predictions(self):

        path = Path(self.config["prediction_file"])

        print("=" * 80)
        print("Prediction file:", path)
        print("Exists:", path.exists())
        print("=" * 80)

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)

        df["Date"] = pd.to_datetime(df["Date"])

        return df

    def run(self):

        if "predictions_df" in self.config:
            predictions = self.config["predictions_df"].copy()
        else:
            predictions = self.load_predictions()

        if "PRED_RETURN" not in predictions.columns:
            raise ValueError(
                "Prediction file does not contain PRED_RETURN. "
                "Run walk-forward training first."
            )

        portfolio = build_portfolio(

            predictions,

            method=self.config.get(
                "portfolio",
                "equal_weight",
            ),

            score_column="PRED_RETURN",

            top_n=self.config["top_n"],

            max_stock_weight=self.config.get(
                "max_stock_weight",
                1.0,
            ),

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

        portfolio["Return"] = VolatilityTarget(
            target_vol=self.config.get(
                "target_volatility",
                0.20,
            ),
        ).apply(
            portfolio["Return"]
        )

        portfolio["Equity"] = (
            1 + portfolio["Return"]
        ).cumprod()

        metrics = evaluate(

            portfolio,

            holding_days=self.config["holding_days"],

        )

        print()

        for k, v in metrics.items():
            print(f"{k} = {v}")

        print()

        return portfolio, metrics
