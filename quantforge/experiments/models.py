from dataclasses import dataclass
from typing import Optional


@dataclass
class Experiment:
    id: str
    created: str
    name: str
    model: str
    portfolio: str
    feature_store: str
    top_n: int
    confidence_quantile: float
    params: str
    git_commit: str
    dataset: str
    status: str


@dataclass
class Metric:
    experiment_id: str
    sharpe: float
    cagr: float
    sortino: float
    calmar: float
    maxdd: float
    turnover: float
    winrate: float
    score: float
