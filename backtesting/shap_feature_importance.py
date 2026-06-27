import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

print("Loading model...")

model = joblib.load(
    "models/monthly_lightgbm_regressor_v12.pkl"
)

print("Loading data...")

df = pd.read_csv(
    "data/training/master_v9.csv"
)

FEATURES = [
    "EMA20","EMA50","EMA200",
    "RSI14","MACD","MACD_SIGNAL","MACD_HIST",
    "ATR14","BB_UPPER","BB_LOWER",
    "VWAP",
    "RETURN_1D",
    "RETURN_5D",
    "LOG_RETURN",
    "RSI14_RANK",
    "RETURN_5D_RANK",
    "Volume_RANK",
    "ATR14_RANK",
    "VOL_20D",
    "BULL_REGIME",
    "HIGH_VOL_REGIME",
    "RETURN_20D",
    "RETURN_60D",
    "RETURN_120D",
    "RETURN_20D_RANK",
    "RETURN_60D_RANK",
    "RETURN_120D_RANK",
    "ATR_PCT",
    "EMA20_OVER_EMA200",
    "VOLUME_RATIO_20D",
    "RETURN_250D",
    "RETURN_250D_RANK",
    "RETURN_20D_MINUS_5D",
    "RETURN_120D_MINUS_20D",
    "PRICE_TO_52W_HIGH"
]

X = (
    df[FEATURES]
    .dropna()
    .sample(
        20000,
        random_state=42
    )
)

print("Building SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("Computing SHAP values...")

shap_values = explainer.shap_values(X)

importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance":
    abs(shap_values).mean(axis=0)

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n====================")
print("SHAP IMPORTANCE")
print("====================")

print(importance)

importance.to_csv(
    "data/checkpoints/shap_importance_v12.csv",
    index=False
)

plt.figure(figsize=(10,8))

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "data/checkpoints/shap_summary_v12.png",
    dpi=300
)

print("\nSaved:")
print("data/checkpoints/shap_importance_v12.csv")
print("data/checkpoints/shap_summary_v12.png")