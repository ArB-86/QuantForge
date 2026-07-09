
import json
from pathlib import Path


class ArtifactManager:

    def __init__(self, context):
        self.context = context

    def save(self):

        exp_dir = Path(
            self.context.artifacts.get("experiment_dir", "")
        )

        if not exp_dir:
            return

        exp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.context.metrics:
            with open(exp_dir / "metrics.json", "w") as f:
                json.dump(
                    self.context.metrics,
                    f,
                    indent=4,
                    default=str,
                )

        with open(exp_dir / "artifacts.json", "w") as f:
            json.dump(
                self.context.artifacts,
                f,
                indent=4,
                default=str,
            )

        with open(exp_dir / "context.json", "w") as f:
            json.dump(
                {
                    "status": self.context.status,
                    "warnings": self.context.warnings,
                    "errors": self.context.errors,
                },
                f,
                indent=4,
                default=str,
            )
