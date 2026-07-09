from quantforge.config.config import Config
from quantforge.training.walkforward import WalkForwardTrainer


def train(config):

    if isinstance(config, str):
        config = Config(config).dict()

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    trainer = WalkForwardTrainer(config)

    return trainer.run()