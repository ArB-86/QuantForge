from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _prepare_dataframe(portfolio: pd.DataFrame, sort_by: list = None) -> pd.DataFrame:
    df = portfolio.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if sort_by is None:
        sort_by = ["Ticker", "Date"] if "Ticker" in df.columns else ["Date"]
    df = df.sort_values(sort_by, kind="mergesort")
    return df


def build_trade_log(
    portfolio: pd.DataFrame,
    return_column: str = "TARGET_20D_RETURN",
    daily_returns_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convert portfolio holdings into a trade log.
    """
    df = portfolio.copy()

    if return_column not in df.columns:
        raise ValueError(f"return_column='{return_column}' not in portfolio. Columns: {df.columns.tolist()}")
    df["Return"] = df[return_column].astype(float)

    required = ["Date", "Ticker", "Weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Portfolio missing columns: {missing}")

    df = _prepare_dataframe(df)

    if daily_returns_df is not None:
        daily = daily_returns_df[["Date", "Ticker", "RETURN_1D"]].copy()
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily_dict = daily.set_index(["Ticker", "Date"]).sort_index()
    else:
        daily_dict = None

    trades: List[Dict] = []
    for ticker, group in df.groupby("Ticker", sort=False):
        group = group.reset_index(drop=True)
        in_trade = False
        start = None
        for i in range(len(group)):
            row = group.iloc[i]
            weight = float(row["Weight"])
            if (not in_trade) and weight > 0:
                in_trade = True
                start = i
                continue

            last_row = (i == len(group) - 1)
            gap = False
            if in_trade and i > 0:
                prev_date = group.iloc[i-1]["Date"]
                if (row["Date"] - prev_date).days > 30:
                    gap = True
            exit_trade = in_trade and (weight <= 0 or gap or last_row)
            if not exit_trade:
                continue

            end = i if last_row else i - 1
            trade = group.iloc[start : end + 1]
            entry = trade.iloc[0]
            exit_ = trade.iloc[-1]

            # PnL Calculation
            if daily_dict is not None:
                try:
                    daily_slice = daily_dict.loc[ticker, entry["Date"]:exit_["Date"]]
                    if isinstance(daily_slice, pd.Series):
                        pnl = float(np.prod(1.0 + daily_slice.values) - 1.0)
                    else:
                        pnl = float(np.prod(1.0 + daily_slice["RETURN_1D"].values) - 1.0)
                except KeyError:
                    pnl = float(trade.iloc[-1]["Return"])
            else:
                pnl = float(trade.iloc[-1]["Return"])

            returns_for_stats = trade["Return"].fillna(0.0).to_numpy(dtype=float)

            trades.append({
                "Ticker": ticker,
                "Entry Date": entry["Date"],
                "Exit Date": exit_["Date"],
                "Entry Weight": float(entry["Weight"]),
                "Exit Weight": float(exit_["Weight"]),
                "Holding Days": len(trade),
                "Net Return": pnl,
                "Average Return": float(np.mean(returns_for_stats)),
                "Volatility": float(np.std(returns_for_stats, ddof=0)),
                "Max Return": float(np.max(returns_for_stats)),
                "Min Return": float(np.min(returns_for_stats)),
                "Turnover": float(np.abs(trade["Weight"].diff().fillna(0)).sum()),
            })

            in_trade = False
            start = None

    trades = pd.DataFrame(trades)
    if len(trades):
        trades = trades.sort_values("Entry Date").reset_index(drop=True)
    return trades


def compute_trade_statistics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    stats = {}
    returns = trades["Net Return"].astype(float)
    winners = trades[returns > 0]
    losers = trades[returns <= 0]

    gross_profit = winners["Net Return"].sum()
    gross_loss = abs(losers["Net Return"].sum()) if len(losers) else 0.0

    stats["Total Trades"] = int(len(trades))
    stats["Winning Trades"] = int(len(winners))
    stats["Losing Trades"] = int(len(losers))
    stats["Win Rate"] = float(len(winners) / len(trades)) if len(trades) else 0.0
    stats["Profit Factor"] = float(gross_profit / gross_loss) if gross_loss else float('inf')
    stats["Expectancy"] = float(returns.mean())
    stats["Avg Winner"] = float(winners["Net Return"].mean()) if len(winners) else 0.0
    stats["Avg Loser"] = float(losers["Net Return"].mean()) if len(losers) else 0.0
    stats["Largest Winner"] = float(returns.max())
    stats["Largest Loser"] = float(returns.min())
    stats["Average Holding Days"] = float(trades["Holding Days"].mean())
    stats["Average Position Size"] = float(trades["Entry Weight"].mean())
    stats["Average Turnover"] = float(trades["Turnover"].mean())

    # Simple worst trade
    stats["Worst Trade"] = float(returns.min())

    rolling_window = min(20, len(trades))
    if rolling_window > 0:
        rolling_sharpe = (returns.rolling(rolling_window).mean() / (returns.rolling(rolling_window).std() + 1e-12)) * np.sqrt(252)
        trades["Rolling Sharpe"] = rolling_sharpe
        stats["Rolling Sharpe Mean"] = float(rolling_sharpe.mean())
        stats["Rolling Sharpe Max"] = float(rolling_sharpe.max())
        stats["Rolling Sharpe Min"] = float(rolling_sharpe.min())
    else:
        stats["Rolling Sharpe Mean"] = 0.0

    stats["Median Winner"] = float(winners["Net Return"].median()) if len(winners) else 0.0
    stats["Median Loser"] = float(losers["Net Return"].median()) if len(losers) else 0.0
    stats["Return Std"] = float(returns.std())
    stats["Return Skew"] = float(returns.skew())
    stats["Return Kurtosis"] = float(returns.kurtosis())

    return stats
