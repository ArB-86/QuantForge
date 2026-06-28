from pathlib import Path
from datetime import datetime
from uuid import uuid4
import json


class ExperimentRegistry:

    def __init__(self):

        self.root = Path("results/experiments")
        self.root.mkdir(
            parents=True,
            exist_ok=True
        )

    def create(self):

        exp_id = (
            "EXP_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid4().hex[:8]
        )

        folder = self.root / exp_id

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        return exp_id, folder

    def save_json(
        self,
        folder,
        name,
        data
    ):

        with open(
            folder / name,
            "w"
        ) as fp:

            json.dump(
                data,
                fp,
                indent=4
            )
