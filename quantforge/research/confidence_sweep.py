import copy
import json
import tempfile
from pathlib import Path

from quantforge.backtest.backtest_engine import BacktestEngine


QUANTILES = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.98,
]


def sweep(config_file):

    config_file = Path(config_file)

    base = json.loads(
        config_file.read_text()
    )

    results = []

    for q in QUANTILES:

        cfg = copy.deepcopy(base)

        cfg["confidence_quantile"] = q

        with tempfile.NamedTemporaryFile(
            suffix=".json",
            delete=False,
            mode="w",
        ) as f:

            json.dump(
                cfg,
                f,
                indent=4,
            )

            temp = f.name

        print()
        print("=" * 80)
        print("Confidence Quantile:", q)
        print("=" * 80)

        _, metrics = BacktestEngine(cfg).run()

        results.append({

            "confidence_quantile": q,

            **metrics,

        })

    return results


if __name__ == "__main__":

    import pandas as pd

    df = pd.DataFrame(

        sweep(
            "configs/lightgbm_regressor.json"
        )

    )

    print()

    print("=" * 80)
    print(df)

    df.to_csv(

        "results/confidence_sweep.csv",

        index=False,

    )

    print()

    print(
        "Saved:",
        "results/confidence_sweep.csv",
    )
