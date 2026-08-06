
import json
from pathlib import Path


class MetricsManager:

    def __init__(self, context):
        self.context = context

    def save(self):

        exp_dir = Path(
            self.context.artifacts["experiment_dir"]
        )

        exp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(
                self.context.metrics,
                f,
                indent=4,
                default=str,
            )
