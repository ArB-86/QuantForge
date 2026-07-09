import json

from pathlib import Path

from quantforge.research.registry import (
    ExperimentRegistry
)


class ExperimentManager:

    def __init__(

        self,

        config_file

    ):

        with open(config_file) as fp:

            self.config = json.load(fp)

        self.registry = ExperimentRegistry()

    def create(self):

        exp_id, folder = self.registry.create()

        self.registry.save_json(

            folder,

            "config.json",

            self.config

        )

        return exp_id, folder
