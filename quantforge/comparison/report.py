from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    experiment: str
    sharpe: float
    cagr: float
    max_drawdown: float
    final_equity: float
    turnover: float
    win_rate: float


def save_result(result, directory="results"):

    Path(directory).mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    path = Path(directory) / filename

    with open(path, "w") as f:
        json.dump(
            result.__dict__,
            f,
            indent=4,
        )

    print(path)
