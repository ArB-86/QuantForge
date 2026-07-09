import pandas as pd


class InformationCoefficient:

    def __init__(
        self,
        prediction_col="PRED_RETURN",
        target_col="TARGET_20D_RETURN",
        date_col="Date",
    ):

        self.prediction_col = prediction_col
        self.target_col = target_col
        self.date_col = date_col

    def daily_ic(self, df):

        return (
            df.groupby(self.date_col)
              .apply(
                  lambda x: x[self.prediction_col].corr(
                      x[self.target_col],
                      method="pearson",
                  )
              )
              .dropna()
        )

    def summary(self, df):

        ic = self.daily_ic(df)

        return {
            "mean_ic": float(ic.mean()),
            "std_ic": float(ic.std()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0,
            "min_ic": float(ic.min()),
            "max_ic": float(ic.max()),
            "positive_days": int((ic > 0).sum()),
            "negative_days": int((ic < 0).sum()),
            "observations": int(len(ic)),
        }
