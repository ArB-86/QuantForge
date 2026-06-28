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

from quantforge.features.store import FeatureStore
from quantforge.features.selector import get_training_features

MODEL = PROJECT_ROOT.parent / "models" / "monthly_lightgbm_regressor_v15.pkl"
DATA = PROJECT_ROOT.parent / "data" / "training" / "master_v9.csv"

print("Loading model...")
model = joblib.load(MODEL)

print("Loading dataset...")
df = pd.read_csv(DATA)

store = FeatureStore()
df = store.build(df)

FEATURES = model.feature_name_

TARGET = "TARGET_20D_RETURN"

df = df.dropna(subset=FEATURES + [TARGET])

print("Rows:", len(df))
print("Features:", len(FEATURES))

sample = df.sample(
    min(20000, len(df)),
    random_state=42
)

print("Running SHAP...")

explainer = shap.TreeExplainer(model)

values = explainer.shap_values(
    sample[list(FEATURES)]
)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": abs(values).mean(axis=0)
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

Path("results").mkdir(
    exist_ok=True
)

importance.to_csv(
    OUT / "shap_importance.csv",
    index=False
)

print()
print(importance.head(30))

print()
print("Saved shap_importance.csv")