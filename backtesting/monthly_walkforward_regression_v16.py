# File: backtesting/monthly_walkforward_regression_v15.py

import os
import pandas as pd
import joblib
from pathlib import Path

from lightgbm import LGBMRegressor

# =====================
# PATHS
# =====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT.parent / "data"

CHECKPOINT_FILE = (
    DATA_ROOT
    / "checkpoints"
    / "monthly_walkforward_regression_v16.csv"
)

MODEL_FILE = (
    PROJECT_ROOT.parent
    / "models"
    / "monthly_lightgbm_regressor_v16.pkl"
)

CHECKPOINT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# =====================
# LOAD DATA
# =====================

print("Loading dataset...")

df = pd.read_csv(
    DATA_ROOT / "training" / "master_v9.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values(["Date", "Ticker"])
    .reset_index(drop=True)
)

# =====================
# TUNE FLAG
# =====================

TUNE = True

# =====================
# PURGE SETTINGS
# =====================

PURGE_DAYS = 5

trading_dates = sorted(df["Date"].unique())

# =====================
# FEATURES
# =====================

FEATURES = [

    # =====================
    # PRICE MOMENTUM
    # =====================

    "RETURN_1D",
    "RETURN_5D",
    "RETURN_20D",
    "RETURN_60D",
    "RETURN_120D",
    "RETURN_250D",
    "LOG_RETURN",

    # =====================
    # CROSS-SECTIONAL RANKS
    # =====================

    "RETURN_5D_RANK",
    "RETURN_20D_RANK",
    "RETURN_60D_RANK",
    "RETURN_120D_RANK",
    "RETURN_250D_RANK",

    "Volume_RANK",
    "ATR14_RANK",
    "SIZE_RANK",

    # =====================
    # VOLATILITY
    # =====================

    "VOL_20D",
    "ATR14",
    "ATR_PCT",

    # =====================
    # TREND
    # =====================

    "EMA20",
    "EMA50",
    "EMA200",
    "EMA20_OVER_EMA200",

    # =====================
    # MOMENTUM INDICATORS
    # =====================

    "RSI14",
    "RSI14_RANK",

    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",

    # =====================
    # VOLUME
    # =====================

    "Volume",
    "DOLLAR_VOLUME",
    "LOG_DOLLAR_VOLUME",
    "VOLUME_RATIO_20D",

    # =====================
    # PRICE LOCATION
    # =====================

    "PRICE_TO_52W_HIGH",

    "BB_UPPER",
    "BB_LOWER",

    "VWAP",

    # =====================
    # MARKET CONTEXT
    # =====================

    "MARKET_RET_20D",

    "RETURN_20D_MINUS_5D",
    "RETURN_120D_MINUS_20D",

    "BULL_REGIME",
    "HIGH_VOL_REGIME"

]

TARGET = "TARGET_10D_RETURN"

df = df.dropna(subset=FEATURES + [TARGET])

# =====================
# MEMORY OPTIMIZATION
# =====================

for c in FEATURES:
    df[c] = df[c].astype("float32")

df[TARGET] = df[TARGET].astype("float32")

print("Dataset Shape:", df.shape)

# =====================
# RESUME SUPPORT
# =====================

completed_months = set()

if CHECKPOINT_FILE.exists():
    old = pd.read_csv(CHECKPOINT_FILE, usecols=["Date"])
    old["Date"] = pd.to_datetime(old["Date"])
    completed_months = set(
        old["Date"].dt.to_period("M").astype(str).unique()
    )
    print("Recovered Months:", len(completed_months))

# =====================
# MONTH LIST
# =====================

start_date = pd.Timestamp("2016-01-01")

months = pd.date_range(
    start=start_date,
    end=df["Date"].max(),
    freq="MS"
)

# =====================
# LOOP
# =====================

model = None

for i in range(36, len(months) - 1):

    train_end = months[i]
    test_start = months[i]
    test_end = months[i + 1]

    month_key = test_start.to_period("M").strftime("%Y-%m")

    if month_key in completed_months:
        print("Skipping:", month_key)
        continue

    # =====================
    # PURGE LOGIC
    # =====================

    train_dates = [d for d in trading_dates if d < train_end]

    if len(train_dates) <= PURGE_DAYS:
        continue

    purge_end = train_dates[-PURGE_DAYS]

    train = df[df["Date"] < purge_end]

    test = df[
        (df["Date"] >= test_start) &
        (df["Date"] < test_end)
    ]

    if len(test) == 0:
        continue

    # =====================
    # TRAIN/VALID SPLIT
    # =====================

    split = int(len(train) * 0.9)
    train_part = train.iloc[:split]
    valid_part = train.iloc[split:]

    X_train = train_part[FEATURES]
    y_train = train_part[TARGET]

    X_valid = valid_part[FEATURES]
    y_valid = valid_part[TARGET]

    X_test = test[FEATURES]

    print("\n====================================")
    print(
        "Training:",
        train["Date"].min().date(),
        "->",
        train["Date"].max().date(),
        "| Purge End:",
        purge_end.date(),
        "| Test:",
        test_start.date(),
        "| Train Rows:",
        len(train),
        "| Test Rows:",
        len(test)
    )

    # =====================
    # FINAL MODEL (fixed hyperparameters from Optuna)
    # =====================

    model = LGBMRegressor(
        device="gpu",
        objective="regression",
        metric="rmse",
        n_estimators=500,
        max_bin=127,
        force_col_wise=True,
        n_jobs=8,
        random_state=42,
        verbose=-1,

        learning_rate=0.013160062941500798,
        num_leaves=184,
        max_depth=5,
        colsample_bytree=0.6693371831367438,
        subsample=0.9189428027008016,
        subsample_freq=1,
        min_child_samples=137,
        reg_alpha=3.6317646149295992,
        reg_lambda=2.601246673004912
    )

    model.fit(train[FEATURES], train[TARGET])
    print("Fit Complete")

    pred_return = model.predict(X_test)
    print("Prediction Complete")

    temp = test.copy()
    temp["PRED_RETURN"] = pred_return

    temp.to_csv(
        CHECKPOINT_FILE,
        mode="a",
        header=not CHECKPOINT_FILE.exists(),
        index=False
    )

    print("Checkpoint Saved:", month_key)

# =====================
# SAVE MODEL
# =====================

if model is not None:
    joblib.dump(model, MODEL_FILE)
    print("\nModel Saved.")

print("\nMonthly Walkforward Regression Complete.")