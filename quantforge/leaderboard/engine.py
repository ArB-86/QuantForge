from quantforge.storage.database.experiments import ExperimentDB


class Leaderboard:

    def __init__(self):

        self.db = ExperimentDB()

    def top(
        self,
        n=20,
    ):

        return self.db.top(n)

    def top_walkforward(
        self,
        n=20,
    ):

        df = self.db.top(n * 5)

        if len(df) == 0:
            return df

        if "walkforward_windows" not in df.columns:
            return df.head(n)

        wf = df[df["walkforward_windows"].notna()].copy()
        if len(wf) == 0:
            return df.head(n)

        wf = wf.sort_values(
            by=["walkforward_best_score", "score"],
            ascending=[False, False],
        )
        return wf.head(n)

    def champion(self):

        rows = self.db.top(1)

        if len(rows):

            return rows[0]

        return None