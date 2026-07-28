import pandas as pd


def compute_attribution(trades: pd.DataFrame):
    """
    Simple performance attribution.

    Returns
    -------
    dict
    """

    if trades.empty:
        return {}

    stats = {}

    stats["Top Contributors"] = (
        trades.groupby("Ticker")["Net Return"]
        .sum()
        .nlargest(10)
        .to_dict()
    )

    stats["Worst Contributors"] = (
        trades.groupby("Ticker")["Net Return"]
        .sum()
        .nsmallest(10)
        .to_dict()
    )

    stats["Average Return By Ticker"] = (
        trades.groupby("Ticker")["Net Return"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .to_dict()
    )

    stats["Trade Count"] = (
        trades.groupby("Ticker")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    return stats
