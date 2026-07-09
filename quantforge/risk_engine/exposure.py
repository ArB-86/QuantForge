class ExposureManager:

    def __init__(
        self,
        gross=1.0,
        net=1.0,
    ):
        self.gross = gross
        self.net = net

    def apply(self, portfolio):

        portfolio = portfolio.copy()

        if "Weight" not in portfolio.columns:
            return portfolio

        for date, g in portfolio.groupby("Date"):

            idx = g.index

            w = g["Weight"]

            gross = w.abs().sum()

            if gross > 0:
                portfolio.loc[idx, "Weight"] = (
                    w / gross * self.gross
                ).values

        return portfolio