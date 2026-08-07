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

        # Ensure required columns exist for the allocator
        if "TARGET_5D" not in predictions.columns:
            predictions["TARGET_5D"] = predictions.get("TARGET_20D_RETURN", predictions.get("RETURN_5D", 0))
        if "RET_1D" not in predictions.columns:
            predictions["RET_1D"] = predictions.get("RETURN_1D", 0)
        if "Raw_Prediction" not in predictions.columns:
            predictions["Raw_Prediction"] = predictions["PRED_RETURN"]
        if "Prediction" not in predictions.columns:
            predictions["Prediction"] = predictions["PRED_RETURN"]

        holdings = build_portfolio(
            predictions,
            method=self.config.get("portfolio", "inverse_vol"),
            score_column="PRED_RETURN",
            top_k=self.config.get("top_n", 10),
            buffer_k=self.config.get("buffer_k", 15),
            rebalance_freq=self.config.get("holding_days", 5),
        )

        print("=" * 80)
        print("Portfolio Method:", self.config["portfolio"])
        print(holdings[["Date", "Ticker", "Weight"]].head(20))
        print("=" * 80)

        # Ensure the target return column exists for the simulator
        if "TARGET_20D_RETURN" not in holdings.columns:
            if "RETURN_20D" in holdings.columns:
                holdings["TARGET_20D_RETURN"] = holdings["RETURN_20D"]
            elif "RETURN_5D" in holdings.columns:
                holdings["TARGET_20D_RETURN"] = holdings["RETURN_5D"]
            else:
                holdings["TARGET_20D_RETURN"] = 0.0

        portfolio = simulate(
            holdings,
            return_column=self.config["target"],
            holding_days=self.config["holding_days"],
            round_trip_cost=self.config["transaction_cost"],
        )

        portfolio["Return"] = VolatilityTarget(
            target_vol=self.config.get("target_volatility", 0.20),
        ).apply(portfolio["Return"])

        portfolio["Equity"] = (1 + portfolio["Return"]).cumprod()

        metrics = evaluate(
            portfolio,
            holding_days=self.config["holding_days"],
        )

        print()
        for k, v in metrics.items():
            print(f"{k} = {v}")
        print()

        return holdings, portfolio, metrics