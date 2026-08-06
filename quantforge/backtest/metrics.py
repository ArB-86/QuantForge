import numpy as np
import pandas as pd


def evaluate(
    portfolio: pd.DataFrame,
    holding_days: int = 5,
):
    """
    Evaluate portfolio performance.

    Returns a dictionary containing performance statistics.

    Expected columns:
        Date
        Return
        Equity

    Optional columns:
        Turnover
        TransactionCost
    """

    if portfolio.empty:
        raise ValueError("Portfolio is empty.")

    required = {"Return", "Equity"}

    missing = required - set(portfolio.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    returns = portfolio["Return"]

    final_equity = float(
        portfolio["Equity"].iloc[-1]
    )

    years = (
        len(portfolio) * holding_days
    ) / 252

    cagr = (
        final_equity ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    volatility = (
        returns.std(ddof=1)
        * np.sqrt(252 / holding_days)
    )

    if volatility > 0:
        sharpe = (
            returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(252 / holding_days)
        )
    else:
        sharpe = np.nan

    downside = returns[returns < 0]

    downside_vol = (
        downside.std(ddof=1)
        * np.sqrt(252 / holding_days)
        if len(downside) > 1
        else np.nan
    )

    if (
        downside_vol is not None
        and not np.isnan(downside_vol)
        and downside_vol > 0
    ):
        sortino = (
            returns.mean()
            / downside.std(ddof=1)
            * np.sqrt(252 / holding_days)
        )
    else:
        sortino = np.nan

    rolling_max = portfolio["Equity"].cummax()

    drawdown = (
        portfolio["Equity"]
        / rolling_max
        - 1
    )

    max_drawdown = drawdown.min()

    if max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)
    else:
        calmar = np.nan

    win_rate = (
        returns > 0
    ).mean()

    stats = {
        "Final Equity": final_equity,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Rebalances": len(portfolio),
        "Average Return": returns.mean(),
        "Best Return": returns.max(),
        "Worst Return": returns.min(),
    }

    if "Turnover" in portfolio.columns:
        stats["Average Turnover"] = (
            portfolio["Turnover"].mean()
        )

    if "TransactionCost" in portfolio.columns:
        stats["Total Transaction Cost"] = (
            portfolio["TransactionCost"].sum()
        )

    return stats