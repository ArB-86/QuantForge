class Pipeline:

    def __init__(self):

        self.steps = []

    def add(self, step):

        self.steps.append(step)

        return self

    def run(self, context):

        for step in self.steps:

            context = step.run(context)

        return context
