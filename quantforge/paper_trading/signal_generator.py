from __future__ import annotations
import pandas as pd

def generate_signals(predictions: pd.DataFrame, top_n: int = 15):
    required = {"Date","Ticker","PRED_RETURN"}
    missing = required - set(predictions.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    latest_date = predictions["Date"].max()
    df = predictions[predictions["Date"]==latest_date].copy()
    df = df.sort_values("PRED_RETURN",ascending=False).head(top_n)
    df["Signal"] = "BUY"
    return df[["Date","Ticker","PRED_RETURN","Signal"]]
