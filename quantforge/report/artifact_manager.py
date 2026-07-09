from pathlib import Path

from quantforge.research.registry import ExperimentRegistry


class ArtifactManager:

    def __init__(self):

        self.registry = ExperimentRegistry()

    def save(
        self,
        config,
        metrics,
        predictions=None,
        model=None,
    ):

        exp_id, folder = self.registry.create()

        self.registry.save_json(
            folder,
            "config.json",
            config,
        )

        self.registry.save_json(
            folder,
            "metrics.json",
            metrics,
        )

        if predictions is not None:
            predictions.to_csv(
                folder / "predictions.csv",
                index=False,
            )

        return {
            "experiment_id": exp_id,
            "folder": str(folder),
        }
