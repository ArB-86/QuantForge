import pandas as pd


class PredictionDiagnostics:

    def __init__(self, df):
        self.df = df.copy()

    def report(self):

        print("=" * 80)
        print("PREDICTION DIAGNOSTICS")
        print("=" * 80)

        print()

        print("Prediction Mean")
        print(self.df["PRED_RETURN"].mean())

        print()

        print("Prediction Std")
        print(self.df["PRED_RETURN"].std())

        print()

        print("Prediction Min")
        print(self.df["PRED_RETURN"].min())

        print()

        print("Prediction Max")
        print(self.df["PRED_RETURN"].max())

        print()

        print("Unique Trading Days")
        print(self.df["Date"].nunique())

        print()

        print("Average Stocks / Day")
        print(
            self.df.groupby("Date")
            .size()
            .mean()
        )

        print()

        daily_ic = (
            self.df
            .groupby("Date")
            .apply(
                lambda x:
                x["PRED_RETURN"].corr(
                    x["TARGET_20D_RETURN"],
                    method="spearman",
                )
            )
        )

        print("Mean Daily IC")
        print(daily_ic.mean())

        print()

        print("Median Daily IC")
        print(daily_ic.median())

        print()

        print("Positive IC %")
        print((daily_ic > 0).mean())
