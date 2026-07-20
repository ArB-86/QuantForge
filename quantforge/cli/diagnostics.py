import json
import pandas as pd

from quantforge.validation.prediction_diagnostics import (
    PredictionDiagnostics,
)


def diagnostics(config):

    with open(config) as f:
        cfg = json.load(f)

    df = pd.read_csv(
        cfg["prediction_file"],
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    PredictionDiagnostics(
        df
    ).report()
