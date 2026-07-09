from quantforge.dataset.loader_builder import DatasetBuilder


class DatasetStage:

    def run(self, context):

        builder = DatasetBuilder(

            context.config["data_path"],

            context.config["features"],

            context.config["target"],

        )

        context.dataset = builder.prepare()

        context.features = builder.features

        return context
