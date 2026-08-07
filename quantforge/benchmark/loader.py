from pathlib import Path
import pandas as pd

def load_benchmark(config):
    path = Path(config["benchmark_file"])
    if not path.exists():
        return None

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    # Fallback: create Return from available columns
    if "Return" not in df.columns:
        if "RETURN_1D" in df.columns:
            df["Return"] = df["RETURN_1D"]
        elif "RETURN_20D" in df.columns:
            df["Return"] = df["RETURN_20D"]
        else:
            df["Return"] = 0.0

    return df.sort_values("Date").reset_index(drop=True)
