import numpy as np


class PortfolioConstraints:

    def __init__(
        self,
        max_weight=0.10,
    ):
        self.max_weight = max_weight

    def apply(self, portfolio):

        portfolio = portfolio.copy()

        if "Weight" not in portfolio.columns:
            return portfolio

        for date, idx in portfolio.groupby("Date").groups.items():

            w = (
                portfolio.loc[idx, "Weight"]
                .to_numpy(dtype=float)
            )

            w = np.minimum(
                w,
                self.max_weight,
            )

            s = w.sum()

            if s > 0:
                w /= s

            portfolio.loc[idx, "Weight"] = w

        return portfolio