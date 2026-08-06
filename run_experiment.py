import sys

from quantforge.core.config.config import Config
from quantforge.research.runner import ExperimentRunner


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python run_experiment.py config.json"
        )

        return

    cfg = Config(
        sys.argv[1]
    )

    runner = ExperimentRunner()

    metrics = runner(
        cfg.dict()
    )

    print()

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    for k, v in metrics.items():

        print(
            f"{k:20}: {v}"
        )


if __name__ == "__main__":

    main()
