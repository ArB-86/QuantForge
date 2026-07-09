import argparse
from pathlib import Path

from quantforge.engine.trainer import train
from quantforge.engine.backtest import backtest
from quantforge.config.config import Config


def main():

    parser = argparse.ArgumentParser(
        prog="QuantForge",
        description="QuantForge Research Platform",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -------------------
    # train
    # -------------------

    p = sub.add_parser("train")
    p.add_argument(
        "--config",
        required=True,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete checkpoint and retrain from scratch",
    )

    # -------------------
    # backtest
    # -------------------

    p = sub.add_parser("backtest")
    p.add_argument(
        "--config",
        required=True,
    )

    # -------------------
    # optimize
    # -------------------

    p = sub.add_parser("optimize")
    p.add_argument(
        "--config",
        required=True,
    )

    # -------------------
    # experiment
    # -------------------

    p = sub.add_parser("experiment")
    p.add_argument(
        "--config",
        required=True,
    )

    # -------------------
    # leaderboard
    # -------------------

    sub.add_parser("leaderboard")

    # -------------------
    # tournament
    # -------------------

    sub.add_parser("tournament")

    args = parser.parse_args()

    if args.command == "train":

        cfg = Config(args.config).dict()

        if args.force:

            ckpt = Path(cfg["checkpoint_file"])
            model = Path(cfg["model_file"])

            if ckpt.exists():
                print("Removing", ckpt)
                ckpt.unlink()

            if model.exists():
                print("Removing", model)
                model.unlink()

        train(args.config)

    elif args.command == "backtest":
        backtest(args.config)

    elif args.command == "optimize":
        print("Not implemented")

    elif args.command == "experiment":
        from quantforge.engine.trainer import train

        metrics = train(args.config)

        print()
        print("=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)

        if isinstance(metrics, dict):
            for k, v in metrics.items():
                print(f"{k:20}: {v}")

    elif args.command == "leaderboard":
        from quantforge.research.leaderboard_engine import LeaderboardEngine

        df = LeaderboardEngine().top(20)

        cols = [
            "id",
            "name",
            "sharpe",
            "cagr",
            "maxdd",
            "winrate",
            "score",
        ]

        # Prepare formatted strings
        header = f"{'Rank':<5} {'ID':<5} {'Experiment':<30} {'Sharpe':<8} {'CAGR':<8} {'DD':<8} {'Win%':<8} {'Score':<8}"
        separator = "=" * len(header)

        print()
        print(separator)
        print(header)
        print(separator)

        for rank, (_, row) in enumerate(df[cols].iterrows(), start=1):
            name = row["name"][:28]  # truncate long names
            print(
                f"{rank:<5} "
                f"{row['id']:<5} "
                f"{name:<30} "
                f"{row['sharpe']:<8.2f} "
                f"{row['cagr']:<8.1%} "
                f"{row['maxdd']:<8.1%} "
                f"{row['winrate']:<8.1%} "
                f"{row['score']:<8.3f}"
            )

        print(separator)

    elif args.command == "tournament":
        print("Not implemented")


if __name__ == "__main__":
    main()