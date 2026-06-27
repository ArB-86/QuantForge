# File: backtesting/monthly_walkforward_ranker_v2.py

import os
import pandas as pd
import joblib

from lightgbm import LGBMRanker

# =====================
# PATHS (updated for ranker v2)
# =====================

CHECKPOINT_FILE = (
    "data/checkpoints/"
    "monthly_walkforward_ranker_v2.csv"
)

MODEL_FILE = (
    "models/monthly_lightgbm_ranker_v2.pkl"
)

os.makedirs("data/checkpoints", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================
# CLEAN OLD ARTIFACTS (safe to run even if files don't exist)
# =====================

for p in [
    CHECKPOINT_FILE,
    MODEL_FILE,
]:
    try:
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        # ignore removal errors
        pass

# =====================
# LOAD DATA
# =====================

print("Loading dataset...")

df = pd.read_csv("data/training/master_v9.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values(["Date", "Ticker"])
    .reset_index(drop=True)
)

# =====================
# PURGE SETTINGS
# =====================

PURGE_DAYS = 5

trading_dates = sorted(df["Date"].unique())

# =====================
# FEATURES
# =====================

FEATURES = [
    "EMA20", "EMA50", "EMA200", "RSI14", "MACD", "MACD_SIGNAL", "MACD_HIST",
    "ATR14", "BB_UPPER", "BB_LOWER", "VWAP", "RETURN_1D", "RETURN_5D",
    "LOG_RETURN", "RSI14_RANK", "RETURN_5D_RANK", "Volume_RANK", "ATR14_RANK",
    "VOL_20D", "BULL_REGIME", "HIGH_VOL_REGIME", "RETURN_20D", "RETURN_60D",
    "RETURN_120D", "RETURN_20D_RANK", "RETURN_60D_RANK", "RETURN_120D_RANK",
    "ATR_PCT", "EMA20_OVER_EMA200", "VOLUME_RATIO_20D", "RETURN_250D",
    "RETURN_250D_RANK", "RETURN_20D_MINUS_5D", "RETURN_120D_MINUS_20D",
    "PRICE_TO_52W_HIGH"
]

TARGET = "TARGET_5D_RETURN"

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

if os.path.exists(CHECKPOINT_FILE):
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
    # BUILD RANK LABELS (deterministic ordering)
    # =====================

    train = train.copy()

    # Sort by Date and TARGET to ensure reproducibility in qcut (ties handled consistently)
    train = train.sort_values(["Date", TARGET], ascending=[True, True])

    train["RANK_LABEL"] = (
        train
        .groupby("Date")[TARGET]
        .transform(
            lambda x: pd.qcut(
                x,
                q=5,
                labels=False,
                duplicates="drop"
            )
        )
    )

    train["RANK_LABEL"] = (
        train["RANK_LABEL"]
        .fillna(2)
        .astype(int)
    )

    # Explicit assertions to catch silent bugs
    assert train["RANK_LABEL"].min() == 0, "Minimum rank label must be 0"
    assert train["RANK_LABEL"].max() == 4, "Maximum rank label must be 4"

    # =====================
    # TRAIN/VALID SPLIT REMOVED (no early stopping used)
    # =====================

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
    # FINAL MODEL (Lambdarank for ranking)
    # =====================

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        device="gpu",
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

    # =====================
    # GROUPS: each trading day is a group (for the entire training set)
    # =====================

    group_train = (
        train
        .groupby("Date")
        .size()
        .tolist()
    )

    # Fit with group information using rank labels
    model.fit(
        train[FEATURES],
        train["RANK_LABEL"],
        group=group_train
    )

    print("Fit Complete")
    # Optional: log rank label distribution for debugging
    print(
        "Rank labels distribution:",
        train["RANK_LABEL"].value_counts().sort_index().to_dict()
    )

    # Prediction (same as regression)
    pred_return = model.predict(X_test)
    print("Prediction Complete")

    temp = test.copy()
    temp["PRED_RETURN"] = pred_return

    temp.to_csv(
        CHECKPOINT_FILE,
        mode="a",
        header=not os.path.exists(CHECKPOINT_FILE),
        index=False
    )

    print("Checkpoint Saved:", month_key)

# =====================
# SAVE MODEL
# =====================

if model is not None:
    # LGBMRanker is a LightGBM model; saving with joblib is fine (.pkl)
    joblib.dump(model, MODEL_FILE)
    print("\nModel Saved.")

print("\nMonthly Walkforward Ranker Complete.")