import glob

from quantforge.core.config.config import Config
from quantforge.research_pipeline.runner import ExperimentRunner


class Tournament:

    def __init__(self):

        self.runner = ExperimentRunner()

    def run(self):

        configs = [

            "configs/lightgbm_regressor.json",

        ]

        results = []

        for cfg_path in configs:

            print("=" * 80)
            print(cfg_path)
            print("=" * 80)

            cfg = Config(cfg_path)

            metrics = self.runner(
                cfg.dict()
            )

            results.append(

                (

                    cfg["name"],

                    metrics["Score"]

                )

            )

        results.sort(

            key=lambda x: x[1],

            reverse=True,

        )

        return results