from pathlib import Path

import pandas as pd

from quantforge.backtest.backtest_engine import BacktestEngine
from quantforge.core.config.config import Config


METHODS = [
    "equal_weight",
    "score_weight",
    "inverse_volatility",
    "risk_parity",
]


def compare(config_path):

    base = Config(config_path).dict()

    rows = []

    for method in METHODS:

        print("=" * 80)
        print(method.upper())
        print("=" * 80)

        cfg = dict(base)
        cfg["portfolio"] = method

        engine = BacktestEngine(cfg)

        _, metrics = engine.run()

        metrics["Method"] = method

        rows.append(metrics)

    df = pd.DataFrame(rows)

    cols = [
        "Method",
        "Final Equity",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Max Drawdown",
        "Volatility",
        "Win Rate",
        "Average Turnover",
        "Total Transaction Cost",
    ]

    df = df[cols]

    Path("results").mkdir(
        exist_ok=True,
    )

    df.to_csv(
        "results/portfolio_comparison.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("PORTFOLIO COMPARISON")
    print("=" * 80)
    print(df)

    return df
