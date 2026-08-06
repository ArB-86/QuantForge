import pandas as pd

class WalkForwardReport:
    def __init__(self):
        self.rows = []

    def add(self, window, metrics):
        row = {
            "Train Start": window.train_start,
            "Train End": window.train_end,
            "Test Start": window.test_start,
            "Test End": window.test_end,
        }
        row.update(metrics)
        self.rows.append(row)

    def dataframe(self):
        return pd.DataFrame(self.rows)

    def summary(self):
        df = self.dataframe()
        numeric = df.select_dtypes("number")
        return numeric.agg(["mean", "median", "std", "min", "max"])
