import pandas as pd


class ExperimentComparison:

    def __init__(
        self,
        metrics_a,
        metrics_b,
    ):

        self.a = metrics_a
        self.b = metrics_b

    def compare(self):

        rows = []

        #
        # Compare only metrics that exist
        # in BOTH experiments.
        #

        metrics = sorted(

            set(self.a.keys())

            &

            set(self.b.keys())

        )

        for metric in metrics:

            a = self.a[metric]
            b = self.b[metric]

            diff = None

            #
            # Compute difference only for numbers
            #

            if isinstance(a, (int, float)) and isinstance(
                b,
                (int, float),
            ):

                diff = b - a

            rows.append(

                {

                    "Metric": metric,

                    "Experiment A": a,

                    "Experiment B": b,

                    "Difference": diff,

                }

            )

        return pd.DataFrame(rows)