from pathlib import Path

import pandas as pd


def load_benchmark(config):
    """
    Load benchmark returns.

    Expected config:

    benchmark_file
    benchmark_return_column
    """

    path = Path(config["benchmark_file"])

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df.sort_values("Date").reset_index(drop=True)
