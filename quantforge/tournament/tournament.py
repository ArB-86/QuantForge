from quantforge.config.config import Config
from quantforge.pipeline.experiment_pipeline import ExperimentPipeline


class Tournament:

    def __init__(self):

        self.pipeline = ExperimentPipeline()

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

            context = self.pipeline.run(
                cfg.dict()
            )

            results.append(

                (

                    cfg["name"],

                    context.metrics["Score"]

                )

            )

        results.sort(

            key=lambda x: x[1],

            reverse=True,

        )

        return results
