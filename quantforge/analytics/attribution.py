from __future__ import annotations

import numpy as np
import pandas as pd


def compute_attribution(
    holdings: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> dict:

    stats = {}

    if portfolio.empty:
        return stats

    returns = portfolio["Return"].astype(float)

    gross_return = float((1.0 + returns).prod() - 1.0)
    net_return = float(portfolio["Equity"].iloc[-1] - 1.0)

    stats["Gross Return"] = gross_return
    stats["Net Return"] = net_return

    if "TransactionCost" in portfolio.columns:
        tc = float(portfolio["TransactionCost"].sum())
    else:
        tc = 0.0

    stats["Transaction Cost"] = tc
    stats["Transaction Cost Drag"] = gross_return - net_return

    if "Weight" in holdings.columns:
        exposure = holdings.groupby("Date")["Weight"].sum()
        stats["Average Exposure"] = float(exposure.mean())
        stats["Max Exposure"] = float(exposure.max())
        stats["Min Exposure"] = float(exposure.min())

        positions = holdings.groupby("Date").size()
        stats["Average Positions"] = float(positions.mean())
        stats["Max Positions"] = int(positions.max())
        stats["Min Positions"] = int(positions.min())
        stats["Average Weight"] = float(holdings["Weight"].mean())

    if "Turnover" in portfolio.columns:
        stats["Average Turnover"] = float(portfolio["Turnover"].mean())
        stats["Total Turnover"] = float(portfolio["Turnover"].sum())

    return stats
