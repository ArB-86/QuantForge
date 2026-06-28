import pandas as pd


class TopNEngine:

    def __init__(self, predictions):

        self.predictions = predictions.copy()

    def evaluate(
        self,
        target,
        top_n,
    ):

        returns = []

        for _, g in self.predictions.groupby("Date"):

            picks = (

                g

                .sort_values(
                    "PRED_RETURN",
                    ascending=False,
                )

                .head(top_n)

            )

            returns.append(

                picks[target].mean()

            )

        return pd.Series(returns)

    def sweep(
        self,
        target,
        top_ns,
    ):

        rows = []

        for n in top_ns:

            r = self.evaluate(
                target,
                n,
            )

            rows.append({

                "TopN": n,

                "Mean": r.mean(),

                "Std": r.std(),

                "WinRate": (
                    (r > 0).mean()
                ),

            })

        return pd.DataFrame(rows)
