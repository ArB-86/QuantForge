import numpy as np
import pandas as pd

from quantforge.portfolio_engine.optimizer import (
    MinimumVarianceOptimizer,
)


def build_risk_parity_portfolio(
    df,
    score_column="PRED_RETURN",
    volatility_column="VOL_20D",
    top_n=10,
    max_weight=0.20,
    min_weight=0.02,
    confidence_quantile=0.90,
    **kwargs,
):
    portfolios = []

    for date, g in df.groupby("Date"):

        # Confidence filter (consistent with score_weight)
        g = g[
            g[score_column] > g[score_column].quantile(confidence_quantile)
        ]

        if len(g) == 0:
            continue

        picks = (
            g.sort_values(
                score_column,
                ascending=False,
            )
            .head(top_n)
            .copy()
        )

        # Build rolling historical return matrix
        history = (
            df[
                (df["Ticker"].isin(picks["Ticker"])) &
                (df["Date"] <= date)
            ][["Date", "Ticker", "RETURN_1D"]]
            .pivot(
                index="Date",
                columns="Ticker",
                values="RETURN_1D",
            )
            .tail(60)
        )

        if len(history) < 20:
            # Fallback to diagonal covariance from volatility
            vol = (
                picks[volatility_column]
                .fillna(picks[volatility_column].median())
                .clip(lower=1e-6)
            )
            covariance = np.diag(np.square(vol.values))
        else:
            covariance = (
                history
                .fillna(0.0)
                .cov()
                .values
            )

        # Ensure no NaN/infinite values
        covariance = np.nan_to_num(
            covariance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        optimizer = MinimumVarianceOptimizer(
            max_weight=max_weight,
            min_weight=min_weight,
        )

        weights = optimizer.optimize(
            covariance
        )

        picks["Weight"] = weights

        portfolios.append(picks)

    return pd.concat(
        portfolios,
        ignore_index=True,
    )
