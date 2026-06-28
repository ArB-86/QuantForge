from quantforge.training.walkforward import WalkForwardTrainer


def train(config):
    """
    Run the walk-forward training pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    CheckpointManager
    """

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    trainer = WalkForwardTrainer(config)

    return trainer.run()