import sqlite3
import pandas as pd  # new import
from pathlib import Path


class ExperimentDB:

    def __init__(self):

        Path("database").mkdir(
            exist_ok=True
        )

        self.conn = sqlite3.connect(
            "database/experiments.db"
        )

        self.conn.execute("""

        CREATE TABLE IF NOT EXISTS experiments(

            id INTEGER PRIMARY KEY,

            name TEXT,

            model TEXT,

            sharpe REAL,

            cagr REAL,

            maxdd REAL,

            winrate REAL,

            score REAL,

            params TEXT,

            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

        """)

        self.conn.commit()

    def insert(
        self,
        row,
    ):

        self.conn.execute(

            """

            INSERT INTO experiments(

                name,
                model,
                sharpe,
                cagr,
                maxdd,
                winrate,
                score,
                params

            )

            VALUES(

                ?,?,?,?,?,?,?,?

            )

            """,

            (

                row["name"],
                row["model"],
                row["Sharpe"],
                row["CAGR"],
                row["Max Drawdown"],
                row["Win Rate"],
                row["Score"],
                row["params"],

            )

        )

        self.conn.commit()

    def top(
        self,
        n=20,
    ):

        rows = self.conn.execute(
            """
            SELECT *
            FROM experiments
            ORDER BY score DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

        columns = [
            "id",
            "name",
            "model",
            "sharpe",
            "cagr",
            "maxdd",
            "winrate",
            "score",
            "params",
            "created",
        ]

        return pd.DataFrame(
            rows,
            columns=columns,
        )
