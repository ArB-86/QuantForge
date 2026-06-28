import pandas as pd


class SHAPEngine:

    def __init__(self, shap_df):

        self.shap_df = shap_df.copy()

    def importance(self):

        cols = [
            c
            for c in self.shap_df.columns
            if c != "Date"
        ]

        out = pd.DataFrame({

            "Feature": cols,

            "mean": [
                self.shap_df[c].abs().mean()
                for c in cols
            ],

            "std": [
                self.shap_df[c].abs().std()
                for c in cols
            ],

        })

        return (

            out

            .sort_values(
                "mean",
                ascending=False,
            )

            .reset_index(drop=True)

        )

    def top_features(
        self,
        n=30,
    ):

        return self.importance().head(n)