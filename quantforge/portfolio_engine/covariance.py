import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf


class CovarianceCache:

    def __init__(
        self,
        df,
        return_column="RETURN_1D",
        lookback=252,
    ):
        self.df = df.copy()

        self.df["Date"] = pd.to_datetime(
            self.df["Date"]
        )

        self.return_column = return_column
        self.lookback = lookback

        self.cache = {}

    def get(
        self,
        date,
        tickers,
    ):

        key = (
            pd.Timestamp(date),
            tuple(sorted(tickers)),
        )

        if key in self.cache:
            return self.cache[key]

        history = self.df[
            (self.df["Date"] <= pd.Timestamp(date))
            &
            (
                self.df["Ticker"].isin(
                    tickers
                )
            )
        ]

        pivot = history.pivot(
            index="Date",
            columns="Ticker",
            values=self.return_column,
        )

        history = pivot.tail(
            self.lookback
        )

        if len(history) < 20:

            cov = np.eye(
                len(tickers)
            )

        else:

            X = history.fillna(
                0.0
            ).values

            cov = (
                LedoitWolf()
                .fit(X)
                .covariance_
            )

            cov = np.nan_to_num(
                cov,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        self.cache[key] = cov

        return cov
