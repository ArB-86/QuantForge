CREATE TABLE IF NOT EXISTS experiments(
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    name TEXT,
    status TEXT,
    created_at TEXT,
    finished_at TEXT,
    config_hash TEXT,
    git_commit TEXT,
    branch TEXT,
    dataset_hash TEXT,
    model_type TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS metrics(
    experiment_id TEXT,
    name TEXT,
    value REAL
);

CREATE TABLE IF NOT EXISTS artifacts(
    experiment_id TEXT,
    name TEXT,
    path TEXT
);

CREATE TABLE IF NOT EXISTS tags(
    experiment_id TEXT,
    tag TEXT
);
