import argparse
import pandas as pd
from pathlib import Path

from quantforge.trainer.engine import train
from quantforge.backtest_engine.engine import backtest
from quantforge.core.config.config import Config
from quantforge.research.shap_analysis import shap_analysis
from quantforge.research.alpha_research import alpha_research


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
    p.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials",
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
    # shap
    # -------------------

    p = sub.add_parser("shap")
    p.add_argument(
        "--config",
        required=True,
    )

    # -------------------
    # compare
    # -------------------

    p = sub.add_parser("compare")
    p.add_argument(
        "--config",
        required=True,
    )

    # -------------------
    # alpha (NEW)
    # -------------------

    p = sub.add_parser("alpha")
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
    sub.add_parser("paper")
    sub.add_parser("live")

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
        from quantforge.research.runner import ExperimentRunner
        from quantforge.automl.engine import OptunaEngine

        cfg = Config(args.config).dict()
        runner = ExperimentRunner()
        engine = OptunaEngine(base_config=cfg, runner=runner)
        engine.optimize(n_trials=args.trials)

    elif args.command == "experiment":
        from quantforge.research_pipeline.runner import ExperimentRunner

        context = ExperimentRunner(args.config).run()

        print()
        print("=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)

        print("Status :", context.status)

        if context.metrics:
            print()
            print("Metrics")
            print("-" * 80)

            for k, v in context.metrics.items():
                print(f"{k:20}: {v}")

        if context.artifacts:
            print()
            print("Artifacts")
            print("-" * 80)
            for k, v in context.artifacts.items():
                print(f"{k:20}: {v}")

    elif args.command == "shap":
        shap_analysis(args.config)

    elif args.command == "compare":
        from quantforge.research.portfolio_comparison import compare
        compare(args.config)

    elif args.command == "alpha":
        cfg = Config(args.config).dict()
        df = pd.read_csv(cfg["checkpoint_file"])
        alpha_research(df)

    elif args.command == "leaderboard":
        from quantforge.leaderboard.engine import Leaderboard

        df = Leaderboard().top(20)

        cols = [
            "id",
            "name",
            "sharpe",
            "cagr",
            "maxdd",
            "winrate",
            "score",
        ]

        header = f"{'Rank':<5} {'ID':<5} {'Experiment':<30} {'Sharpe':<8} {'CAGR':<8} {'DD':<8} {'Win%':<8} {'Score':<8}"
        separator = "=" * len(header)

        print()
        print(separator)
        print(header)
        print(separator)

        for rank, (_, row) in enumerate(df[cols].iterrows(), start=1):
            name = row["name"][:28]
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
        from quantforge.tournament.tournament import Tournament

        results = Tournament().run()

        print("\n" + "=" * 80)
        print("TOURNAMENT RESULTS")
        print("=" * 80)

        for i, (name, score) in enumerate(results, 1):
            print(f"{i:2d}. {name:<40} {score:.6f}")

if __name__ == "__main__":
    main()
