import pandas as pd


class PredictionEngine:

    def __init__(self, predictions):

        self.predictions = predictions.copy()

    def correlation(self, target):

        return self.predictions[
            "PRED_RETURN"
        ].corr(
            self.predictions[target]
        )

    def top_n_return(
        self,
        target,
        n,
    ):

        rows = []

        for _, g in self.predictions.groupby("Date"):

            rows.append(

                g
                .sort_values(
                    "PRED_RETURN",
                    ascending=False,
                )
                .head(n)[target]
                .mean()

            )

        return pd.Series(rows)

    def deciles(
        self,
        target,
    ):

        df = self.predictions.copy()

        df["DECILE"] = pd.qcut(

            df["PRED_RETURN"],

            10,

            labels=False,

            duplicates="drop",

        )

        return (

            df
            .groupby("DECILE")[target]
            .agg(
                [
                    "mean",
                    "std",
                    "count",
                ]
            )

        )

    def summary(self):

        return self.predictions[
            "PRED_RETURN"
        ].describe()
