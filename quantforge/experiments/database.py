import sqlite3
from pathlib import Path


class ExperimentDatabase:

    def __init__(
        self,
        db_path="database/experiments_v2.db",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.conn = sqlite3.connect(
            self.db_path
        )

        self.conn.execute(
            "PRAGMA foreign_keys=ON;"
        )

        self.create_schema()

    def create_schema(self):

        self.conn.executescript(
            '''
CREATE TABLE IF NOT EXISTS experiments(

id TEXT PRIMARY KEY,

created TEXT NOT NULL,

name TEXT,

model TEXT,

portfolio TEXT,

feature_store TEXT,

top_n INTEGER,

confidence_quantile REAL,

params TEXT,

git_commit TEXT,

dataset TEXT,

status TEXT

);

CREATE TABLE IF NOT EXISTS metrics(

experiment_id TEXT PRIMARY KEY,

sharpe REAL,

cagr REAL,

sortino REAL,

calmar REAL,

maxdd REAL,

turnover REAL,

winrate REAL,

score REAL,

FOREIGN KEY(experiment_id)
REFERENCES experiments(id)
ON DELETE CASCADE

);

CREATE TABLE IF NOT EXISTS artifacts(

id INTEGER PRIMARY KEY AUTOINCREMENT,

experiment_id TEXT,

kind TEXT,

path TEXT,

FOREIGN KEY(experiment_id)
REFERENCES experiments(id)
ON DELETE CASCADE

);

CREATE INDEX IF NOT EXISTS idx_model
ON experiments(model);

CREATE INDEX IF NOT EXISTS idx_score
ON metrics(score DESC);
'''
        )

        self.conn.commit()

    def tables(self):
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [r[0] for r in cur.fetchall()]

    def close(self):
        self.conn.close()
