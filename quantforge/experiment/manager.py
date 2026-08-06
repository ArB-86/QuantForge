import json
from pathlib import Path

from quantforge.registry.database import ExperimentRegistry


class ExperimentManager:

    def __init__(self, config_file):
        with open(config_file) as fp:
            self.config = json.load(fp)
        self.registry = ExperimentRegistry()

    def create(self, folder=None):
        """
        Create a new experiment entry.

        Args:
            folder: Optional folder path. If not provided, the registry generates one.
        Returns:
            (experiment_id, folder_path)
        """
        if folder is None:
            # Let the registry create the folder and return the path
            folder_path = self.registry.create()
            exp_id = folder_path.name  # or whatever ID the registry uses
        else:
            folder_path = Path(folder)
            folder_path.mkdir(parents=True, exist_ok=True)
            exp_id = self.registry.create_from_path(str(folder_path))  # if such method exists

        # Save the config in the folder
        self.registry.save_json(folder_path, "config.json", self.config)

        return exp_id, folder_path
