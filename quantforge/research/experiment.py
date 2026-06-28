
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime


@dataclass
class Experiment:

    name: str

    model: str

    features: int

    sharpe: float

    cagr: float

    max_drawdown: float

    win_rate: float

    turnover: float

    notes: str = ""


class ExperimentTracker:

    def __init__(
        self,
        folder="results"
    ):

        self.folder = Path(folder)

        self.folder.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(
        self,
        exp
    ):

        filename = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + exp.name
            + ".json"
        )

        with open(
            self.folder / filename,
            "w"
        ) as f:

            json.dump(
                exp.__dict__,
                f,
                indent=4
            )

        print(
            "Saved:",
            filename
        )
