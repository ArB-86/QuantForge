from pathlib import Path
import pandas as pd


def load_benchmark(config):
    """
    Load benchmark returns if configured.
    Returns None when benchmark is disabled (no 'benchmark_file' key).
    """
    benchmark_file = config.get("benchmark_file")
    if not benchmark_file:
        return None

    path = Path(benchmark_file)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)
