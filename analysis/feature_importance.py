import joblib
import pandas as pd

MODEL = "models/monthly_lightgbm_regressor_v12.pkl"

FEATURES = [
    "EMA20","EMA50","EMA200","RSI14","MACD","MACD_SIGNAL",
    "MACD_HIST","ATR14","BB_UPPER","BB_LOWER","VWAP",
    "RETURN_1D","RETURN_5D","LOG_RETURN","RSI14_RANK",
    "RETURN_5D_RANK","Volume_RANK","ATR14_RANK",
    "VOL_20D","BULL_REGIME","HIGH_VOL_REGIME",
    "RETURN_20D","RETURN_60D","RETURN_120D",
    "RETURN_20D_RANK","RETURN_60D_RANK",
    "RETURN_120D_RANK","ATR_PCT",
    "EMA20_OVER_EMA200",
    "VOLUME_RATIO_20D",
    "RETURN_250D",
    "RETURN_250D_RANK",
    "RETURN_20D_MINUS_5D",
    "RETURN_120D_MINUS_20D",
    "PRICE_TO_52W_HIGH"
]

model = joblib.load(MODEL)

imp = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

imp = imp.sort_values(
    "Importance",
    ascending=False
)

print(imp)

imp.to_csv(
    "analysis/feature_importance.csv",
    index=False
)

print()
print("Saved: analysis/feature_importance.csv")
