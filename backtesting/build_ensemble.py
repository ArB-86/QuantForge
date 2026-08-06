import pandas as pd

# ====================
# LOAD
# ====================

lgb = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression.csv"
)

cat = pd.read_csv(
    "data/checkpoints/monthly_walkforward_catboost.csv"
)

# ====================
# KEEP ONLY NEEDED
# ====================

cat = cat[
    [
        "Date",
        "Ticker",
        "PRED_RETURN"
    ]
].rename(
    columns={
        "PRED_RETURN":
        "CATBOOST_PRED"
    }
)

# ====================
# MERGE
# ====================

df = lgb.merge(
    cat,
    on=[
        "Date",
        "Ticker"
    ],
    how="inner"
)

# ====================
# ENSEMBLE
# ====================

df["ENSEMBLE_PRED"] = (
    df["PRED_RETURN"]
    +
    df["CATBOOST_PRED"]
) / 2

# ====================
# SAVE
# ====================

df.to_csv(
    "data/checkpoints/monthly_walkforward_ensemble.csv",
    index=False
)

print(
    "Rows:",
    len(df)
)

print(
    "Saved Ensemble File"
)
