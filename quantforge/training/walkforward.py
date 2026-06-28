from quantforge.training.walkforward_legacy import run_walkforward

class WalkForwardTrainer:

    def __init__(self, config):

        self.config = config

    def run(self):

        return run_walkforward(self.config)
