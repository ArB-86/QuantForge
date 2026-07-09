from quantforge.config.loader import load
from quantforge.training.walkforward import WalkForwardTrainer

def run(config_path):

    config = load(config_path)

    engine = WalkForwardTrainer(config)

    return engine
