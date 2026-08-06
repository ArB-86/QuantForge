import numpy as np
import pandas as pd

from quantforge.portfolio_engine.optimizer import MinimumVarianceOptimizer
from quantforge.portfolio_engine.covariance import CovarianceCache


def build_risk_parity_portfolio(
    df,
    score_column="PRED_RETURN",
    volatility_column="VOL_20D",
    top_n=10,
    max_weight=0.20,
    min_weight=0.02,
    confidence_quantile=0.80,
    return_column="RETURN_1D",
    lookback=252,
    blend=0.35,
    **kwargs,
):
    portfolios = []

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    cov_cache = CovarianceCache(
        df,
        return_column=return_column,
        lookback=lookback,
    )

    optimizer = MinimumVarianceOptimizer(
        max_weight=max_weight,
        min_weight=min_weight,
    )

    dates = np.sort(df["Date"].unique())

    global_vol_median = df[volatility_column].median()
    if np.isnan(global_vol_median):
        global_vol_median = 0.01

    for i, date in enumerate(dates):
        current = df[df["Date"] == date].copy()
        if len(current) == 0:
            continue

        current = current[current[score_column] > current[score_column].quantile(confidence_quantile)]
        if len(current) == 0:
            continue

        picks = current.sort_values(score_column, ascending=False).head(top_n).copy()
        n = len(picks)
        if n == 0:
            continue

        tickers = picks["Ticker"].tolist()

        cov_matrix = cov_cache.get(date, tickers)

        # ---- Safe median: handle all-NaN gracefully ----
        vol_values = picks[volatility_column].to_numpy(dtype=float)
        if np.isnan(vol_values).all():
            vol_median = 1.0
        else:
            vol_median = np.nanmedian(vol_values)
            if np.isnan(vol_median):
                vol_median = global_vol_median

        if cov_matrix.shape != (n, n):
            sigma = picks[volatility_column].fillna(vol_median).clip(lower=1e-6).to_numpy(dtype=float)
            corr = np.full((n, n), 0.15)
            np.fill_diagonal(corr, 1.0)
            cov_matrix = np.outer(sigma, sigma) * corr

        weights = optimizer.optimize(cov_matrix)

        scores = picks[score_column].clip(lower=0)
        vol = picks[volatility_column].fillna(vol_median).clip(lower=1e-6)

        alpha = scores / vol
        if alpha.sum() > 0:
            alpha = alpha / alpha.sum()
        else:
            alpha = np.repeat(1 / n, n)

        weights = blend * weights + (1.0 - blend) * alpha
        weights = weights / weights.sum()
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()

        picks["Weight"] = weights
        portfolios.append(picks)

    if portfolios:
        return pd.concat(portfolios, ignore_index=True)
    else:
        return pd.DataFrame()
