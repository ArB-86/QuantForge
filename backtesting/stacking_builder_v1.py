
import joblib
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data" / "checkpoints"
MODELS = ROOT.parent / "models"

df = pd.read_csv(
    DATA / "monthly_walkforward_ensemble_v3.csv"
)

FEATURES = [
    "PRED_RETURN_5",
    "PRED_RETURN_10",
    "PRED_RETURN_20",
]

TARGET = "TARGET_20D_RETURN"

model = LGBMRegressor(

    device="gpu",

    objective="regression",

    metric="rmse",

    n_estimators=300,

    learning_rate=0.03,

    num_leaves=31,

    force_col_wise=True,

    random_state=42,

    verbose=-1,
)

model.fit(
    df[FEATURES],
    df[TARGET]
)

df["STACK_SCORE"] = model.predict(
    df[FEATURES]
)

MODELS.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    model,
    MODELS / "stacking_v1.pkl"
)

df.to_csv(
    DATA / "monthly_walkforward_stacking_v1.csv",
    index=False
)

print()
print(df[FEATURES + ["STACK_SCORE"]].head())
print()
print("Saved:", len(df))
