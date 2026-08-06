import os
import json
import datetime
import pandas as pd


EXPERIMENT_DIR = "experiments/history"

os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def save_experiment(
    name,
    config,
    metrics
):

    ts = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    folder = os.path.join(
        EXPERIMENT_DIR,
        ts + "_" + name
    )

    os.makedirs(folder, exist_ok=True)

    with open(
        os.path.join(folder, "config.json"),
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    with open(
        os.path.join(folder, "metrics.json"),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    leaderboard = os.path.join(
        EXPERIMENT_DIR,
        "leaderboard.csv"
    )

    row = {
        "Timestamp": ts,
        "Experiment": name,
        **metrics
    }

    if os.path.exists(leaderboard):

        df = pd.read_csv(
            leaderboard
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )

    else:

        df = pd.DataFrame(
            [row]
        )

    df.to_csv(
        leaderboard,
        index=False
    )

    print()

    print("Saved:", folder)
