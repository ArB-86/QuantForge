from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pandas as pd


class ExperimentDB:
    def __init__(self):
        Path("database").mkdir(exist_ok=True)

        self.conn = sqlite3.connect(
            "database/experiments.db",
            check_same_thread=False,
        )

        self.conn.execute(
            """
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
                study_name TEXT,
                trial_count INTEGER,
                best_trial_number INTEGER,
                best_params TEXT,
                best_value REAL,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()
        self.lock = threading.Lock()

    def _ensure_columns(self):
        existing = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        additions = {
            "study_name": "TEXT",
            "trial_count": "INTEGER",
            "best_trial_number": "INTEGER",
            "best_params": "TEXT",
            "best_value": "REAL",
        }
        for col, ddl in additions.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {ddl}")
        self.conn.commit()

    def insert(self, row):
        with self.lock:
            self._ensure_columns()

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
                    params,
                    study_name,
                    trial_count,
                    best_trial_number,
                    best_params,
                    best_value
                )
                VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    row.get("name"),
                    row.get("model"),
                    row.get("Sharpe"),
                    row.get("CAGR"),
                    row.get("Max Drawdown"),
                    row.get("Win Rate"),
                    row.get("Score"),
                    row.get("params"),
                    row.get("study_name"),
                    row.get("trial_count"),
                    row.get("best_trial_number"),
                    row.get("best_params"),
                    row.get("best_value"),
                ),
            )
            self.conn.commit()

    def top(self, n=20):
        self._ensure_columns()
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
            "study_name",
            "trial_count",
            "best_trial_number",
            "best_params",
            "best_value",
            "created",
        ]

        return pd.DataFrame(rows, columns=columns)
