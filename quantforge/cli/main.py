import argparse
from pathlib import Path

from quantforge.config.config import Config
from quantforge.pipeline.experiment_pipeline import run as run_pipeline
from quantforge.research.leaderboard_engine import LeaderboardEngine


def main():

    parser = argparse.ArgumentParser(
        prog="QuantForge",
        description="QuantForge Research Platform",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p = sub.add_parser("experiment")
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true")

    sub.add_parser("leaderboard")

    args = parser.parse_args()

    if args.command == "experiment":

        cfg = Config(args.config).dict()

        if args.force:

            for key in (
                "checkpoint_file",
                "model_file",
            ):

                f = Path(cfg[key])

                if f.exists():
                    print("Removing", f)
                    f.unlink()

        context = run_pipeline(args.config)

        print()
        print("=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)

        for k, v in context.metrics.items():
            print(f"{k:20}: {v}")

    elif args.command == "leaderboard":

        df = LeaderboardEngine().top(20)

        print(df)


if __name__ == "__main__":
    main()
