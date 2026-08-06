import pandas as pd

# =====================================
# LOAD PREDICTIONS
# =====================================

pred5 = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression.csv"
)

pred10 = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression_v13.csv"
)

pred20 = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression_v14.csv"
)

# =====================================
# KEEP REQUIRED COLUMNS
# (Preserve risk features)
# =====================================

pred5 = pred5[
    [
        "Date",
        "Ticker",

        "VOL_20D",
        "MARKET_RET_20D",

        "TARGET_5D_RETURN",
        "TARGET_10D_RETURN",
        "TARGET_20D_RETURN",

        "PRED_RETURN"
    ]
].rename(
    columns={
        "PRED_RETURN": "PRED5"
    }
)

pred10 = pred10[
    [
        "Date",
        "Ticker",
        "PRED_RETURN"
    ]
].rename(
    columns={
        "PRED_RETURN": "PRED10"
    }
)

pred20 = pred20[
    [
        "Date",
        "Ticker",
        "PRED_RETURN"
    ]
].rename(
    columns={
        "PRED_RETURN": "PRED20"
    }
)

# =====================================
# MERGE
# =====================================

df = pred5.merge(
    pred10,
    on=[
        "Date",
        "Ticker"
    ],
    how="inner"
)

df = df.merge(
    pred20,
    on=[
        "Date",
        "Ticker"
    ],
    how="inner"
)

print(df.shape)

# =====================================
# DAILY RANKS
# =====================================

df["RANK5"] = (
    df.groupby("Date")["PRED5"]
      .rank(pct=True)
)

df["RANK10"] = (
    df.groupby("Date")["PRED10"]
      .rank(pct=True)
)

df["RANK20"] = (
    df.groupby("Date")["PRED20"]
      .rank(pct=True)
)

# =====================================
# ENSEMBLE SCORE
# =====================================

df["ENSEMBLE_SCORE"] = (

      0.50 * df["RANK5"]

    + 0.30 * df["RANK10"]

    + 0.20 * df["RANK20"]

)

# =====================================
# SAVE
# =====================================

OUTPUT_FILE = (
    "data/checkpoints/"
    "monthly_walkforward_ensemble_v1.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()

print("====================================")
print("ENSEMBLE CREATED")
print("====================================")

print()

print(df.head())

print()

print(df.columns.tolist())

print()

print("Saved to:", OUTPUT_FILE)