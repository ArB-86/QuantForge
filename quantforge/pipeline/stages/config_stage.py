from quantforge.core.config.loader import load


class ConfigStage:

    def run(self, context):

        if isinstance(context.config, str):

            context.config = load(context.config)

        return context
