import sqlite3

import pandas as pd


class LeaderboardEngine:

    def __init__(
        self,
        database="database/experiments.db",
    ):

        self.database = database

    def dataframe(self):

        conn = sqlite3.connect(
            self.database
        )

        df = pd.read_sql(
            """
            SELECT *
            FROM experiments
            ORDER BY score DESC
            """,
            conn,
        )

        conn.close()

        return df

    def champion(self):

        df = self.dataframe()

        if df.empty:

            return None

        return df.iloc[0]

    def get(
        self,
        experiment_id,
    ):

        df = self.dataframe()

        rows = df[
            df["id"] == experiment_id
        ]

        if rows.empty:

            raise ValueError(
                f"Experiment {experiment_id} not found."
            )

        return rows.iloc[0]

    def top(
        self,
        n=10,
    ):

        return self.dataframe().head(n)