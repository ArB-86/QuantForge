import os
import pandas as pd
import joblib
from lightgbm import LGBMClassifier

# =====================
# PATHS
# =====================

CHECKPOINT_FILE = (
    "data/checkpoints/"
    "monthly_walkforward_predictions.csv"
)

MODEL_FILE = (
    "models/monthly_lightgbm.pkl"
)

os.makedirs(
    "data/checkpoints",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)

# =====================
# LOAD DATA
# =====================

print("Loading dataset...")

df = pd.read_csv(
    "data/training/master_v5.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = (
    df
    .sort_values(
        ["Date", "Ticker"]
    )
    .reset_index(
        drop=True
    )
)

# =====================
# FEATURES
# =====================

FEATURES = [
    "EMA20",
    "EMA50",
    "EMA200",
    "RSI14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "ATR14",
    "BB_UPPER",
    "BB_LOWER",
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
    "HIGH_VOL_REGIME"
]

TARGET = "TARGET_5D_CLASS"

df = df.dropna(
    subset=FEATURES + [TARGET]
)

# =====================
# MEMORY OPTIMIZATION
# =====================

for c in FEATURES:
    df[c] = df[c].astype(
        "float32"
    )

df[TARGET] = df[TARGET].astype(
    "int8"
)

print(
    "Dataset Shape:",
    df.shape
)

# =====================
# RESUME SUPPORT
# =====================

completed_months = set()

if os.path.exists(
    CHECKPOINT_FILE
):
    old = pd.read_csv(
        CHECKPOINT_FILE,
        usecols=["Date"]
    )

    old["Date"] = pd.to_datetime(
        old["Date"]
    )

    completed_months = set(
        old["Date"]
        .dt.to_period("M")
        .astype(str)
        .unique()
    )

    print(
        "Recovered Months:",
        len(completed_months)
    )

# =====================
# MONTH LIST
# =====================

start_date = pd.Timestamp(
    "2016-01-01"
)

months = pd.date_range(
    start=start_date,
    end=df["Date"].max(),
    freq="MS"
)

# =====================
# LOOP
# =====================

model = None

for i in range(
    36,
    len(months) - 1
):

    train_end = months[i]

    test_start = months[i]
    test_end = months[i + 1]

    month_key = (
        test_start
        .to_period("M")
        .strftime("%Y-%m")
    )

    if month_key in completed_months:

        print(
            "Skipping:",
            month_key
        )

        continue

    train = df[
        df["Date"] < train_end
    ]

    test = df[
        (df["Date"] >= test_start)
        &
        (df["Date"] < test_end)
    ]

    if len(test) == 0:
        continue

    X_train = train[
        FEATURES
    ]

    y_train = train[
        TARGET
    ]

    X_test = test[
        FEATURES
    ]

    print(
        "\n===================================="
    )

    print(
        "Training:",
        train["Date"].min().date(),
        "->",
        train["Date"].max().date(),
        "| Test:",
        test_start.date(),
        "| Train Rows:",
        len(train),
        "| Test Rows:",
        len(test)
    )

    model = LGBMClassifier(
        device="gpu",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_bin=127,
        subsample=0.8,
        colsample_bytree=0.8,
        force_col_wise=True,
        random_state=42,
        n_jobs=32,
        verbose=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Fit Complete"
    )

    pred = model.predict(
        X_test
    )

    proba = model.predict_proba(
        X_test
    )

    print(
        "Prediction Complete"
    )

    temp = test.copy()

    temp["PRED_CLASS"] = pred
    temp["PROBA_DOWN"] = proba[:, 0]
    temp["PROBA_NEUTRAL"] = proba[:, 1]
    temp["PROBA_UP"] = proba[:, 2]

    temp.to_csv(
        CHECKPOINT_FILE,
        mode="a",
        header=not os.path.exists(
            CHECKPOINT_FILE
        ),
        index=False
    )

    print(
        "Checkpoint Saved:",
        month_key
    )

# =====================
# SAVE MODEL
# =====================

if model is not None:

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\nModel Saved."
    )

print(
    "\nMonthly Walkforward Complete."
)
