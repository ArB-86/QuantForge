import pandas as pd


class RankInformationCoefficient:

    def __init__(
        self,
        prediction_col="PRED_RETURN",
        target_col="TARGET_20D_RETURN",
        date_col="Date",
    ):

        self.prediction_col = prediction_col
        self.target_col = target_col
        self.date_col = date_col

    def daily_rank_ic(self, df):

        return (
            df.groupby(self.date_col)
              .apply(
                  lambda x: x[self.prediction_col].corr(
                      x[self.target_col],
                      method="spearman",
                  )
              )
              .dropna()
        )

    def summary(self, df):

        ric = self.daily_rank_ic(df)

        return {
            "mean_rank_ic": float(ric.mean()),
            "std_rank_ic": float(ric.std()),
            "rank_icir": float(ric.mean() / ric.std()) if ric.std() > 0 else 0.0,
            "min_rank_ic": float(ric.min()),
            "max_rank_ic": float(ric.max()),
            "positive_days": int((ric > 0).sum()),
            "negative_days": int((ric < 0).sum()),
            "observations": int(len(ric)),
        }
