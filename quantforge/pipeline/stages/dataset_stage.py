from quantforge.training.dataset_builder import DatasetBuilder


class DatasetStage:

    def run(self, context):

        cfg = context.config

        builder = DatasetBuilder(
            cfg["data_path"],
            cfg["features"],
            cfg["target"],
        )

        context.dataset = builder.prepare()
        context.features = builder.features

        return context
