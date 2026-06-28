import os
from pathlib import Path
OUT = Path(os.environ["QF_EXPERIMENT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import shap
import pandas as pd
import numpy as np

from quantforge.features.store import FeatureStore
from quantforge.features.selector import get_training_features

MODEL = PROJECT_ROOT.parent / "models" / "monthly_lightgbm_regressor_v15.pkl"
DATA = PROJECT_ROOT.parent / "data" / "training" / "master_v9.csv"

print("Loading model...")
model = joblib.load(MODEL)

print("Loading data...")
df = pd.read_csv(DATA)

df["Date"] = pd.to_datetime(df["Date"])

store = FeatureStore()
df = store.build(df)

FEATURES = list(model.feature_name_)

TARGET = "TARGET_20D_RETURN"

df = df.dropna(subset=FEATURES + [TARGET])

months = sorted(
    df["Date"].dt.to_period("M").unique()
)

results = []

explainer = shap.TreeExplainer(model)

for month in months:

    sample = df[
        df["Date"].dt.to_period("M") == month
    ]

    if len(sample) > 2000:

        sample = sample.sample(
            2000,
            random_state=42
        )

    values = explainer.shap_values(
        sample[FEATURES]
    )

    imp = np.abs(values).mean(axis=0)

    temp = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": imp,

        "Month": str(month)

    })

    results.append(temp)

    print(month)

final = pd.concat(results)

final.to_csv(
    OUT / "shap_walkforward.csv",
    index=False
)

summary = (
    final
    .groupby("Feature")["Importance"]
    .agg(["mean","std"])
    .sort_values(
        "mean",
        ascending=False
    )
)

summary.to_csv(
    OUT / "shap_walkforward_summary.csv"
)

print()
print(summary.head(30))
print()
print("Done.")
