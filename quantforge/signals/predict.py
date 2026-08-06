import joblib
import pandas as pd


class PredictionEngine:
    """
    Production prediction wrapper.

    Input:
        Feature DataFrame

    Output:
        ticker
        close
        score
        probability_up
        expected_return
    """

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, df):

        meta = df[["date", "ticker", "close"]].copy()

        X = df.drop(
            columns=[
                "date",
                "ticker",
                "close",
            ],
            errors="ignore",
        )

        #
        # Regression
        #

        pred = self.model.predict(X)

        meta["expected_return"] = pred

        #
        # Score
        #

        meta["score"] = (
            meta["expected_return"]
            .rank(pct=True)
        )

        #
        # Probability
        #

        mn = meta.expected_return.min()
        mx = meta.expected_return.max()

        if mx > mn:

            meta["probability_up"] = (
                meta.expected_return - mn
            ) / (mx - mn)

        else:

            meta["probability_up"] = 0.5

        return meta
