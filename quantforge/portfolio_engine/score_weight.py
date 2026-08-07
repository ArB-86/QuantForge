import numpy as np
import pandas as pd

def build_score_weight_portfolio(
    df,
    score_column="Prediction",
    top_n=10,
    max_weight=0.20,
    min_weight=0.02,
    confidence_quantile=0.90,
):
    portfolios = []

    for date, g in df.groupby("Date"):
        # Conditional Liquidity filter
        if "LOG_DOLLAR_VOLUME" in g.columns:
            g = g[g["LOG_DOLLAR_VOLUME"] >= g["LOG_DOLLAR_VOLUME"].quantile(0.30)]

        # Conditional ATR filter
        if "ATR_PCT" in g.columns:
            g = g[g["ATR_PCT"] <= 0.08]

        if len(g) == 0:
            continue

        # Confidence Filter
        g = g[g[score_column] > g[score_column].quantile(confidence_quantile)]

        if len(g) == 0:
            continue

        picks = g.sort_values(score_column, ascending=False).head(top_n).copy()
        scores = picks[score_column].clip(lower=0)
        
        # Volatility-adjusted scoring
        if "VOL_20D" in picks.columns:
            vol = picks["VOL_20D"].fillna(picks["VOL_20D"].median()).clip(lower=1e-6)
            scores = scores / vol

        if scores.sum() == 0:
            weights = np.repeat(1 / len(picks), len(picks))
        else:
            weights = scores / scores.sum()

        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()

        picks["Weight"] = weights
        portfolios.append(picks)

    if not portfolios:
        return pd.DataFrame()
        
    return pd.concat(portfolios, ignore_index=True)
