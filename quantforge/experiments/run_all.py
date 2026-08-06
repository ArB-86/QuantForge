from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import time

from quantforge.experiments.runner import ExperimentRunner


CONFIG_DIR = Path("configs")


def run_one(config_path):

    runner = ExperimentRunner(config_path)

    metrics = runner.run()

    return (
        Path(config_path).name,
        metrics,
    )


def main():

    # Filter: only configs with prediction_file (backtest configs)
    configs = []

    for cfg in sorted(CONFIG_DIR.glob("*.json")):

        with open(cfg) as f:
            data = json.load(f)

        if "prediction_file" in data:
            configs.append(cfg)

    print("=" * 80)
    print(f"{len(configs)} backtest configs found")
    print(f"Total CPU cores: {os.cpu_count()}")

    # Cap workers to avoid excessive memory pressure
    max_workers = min(16, len(configs), os.cpu_count())
    print(f"Using {max_workers} workers")
    print("=" * 80)

    if not configs:
        print("No valid backtest configs found.")
        return

    start = time.time()

    results = []

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as pool:

        futures = [
            pool.submit(
                run_one,
                str(cfg),
            )
            for cfg in configs
        ]

        for future in as_completed(
            futures
        ):

            name, metrics = future.result()

            print(
                f"✓ {name}"
            )

            results.append(
                {
                    "config": name,
                    **metrics,
                }
            )

    Path("results").mkdir(
        exist_ok=True
    )

    with open(
        "results/leaderboard.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    elapsed = time.time() - start

    print()
    print("=" * 80)
    print(f"Finished in {elapsed:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":

    main()
