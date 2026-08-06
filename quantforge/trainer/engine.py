from quantforge.core.config.config import Config
from quantforge.training.walkforward import WalkForwardTrainer


def train(config, skip_feature_importance=False, dashboard=None):
    if isinstance(config, str):
        config = Config(config).dict()

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    trainer = WalkForwardTrainer(config, skip_feature_importance=skip_feature_importance, dashboard=dashboard)
    return trainer.run()
