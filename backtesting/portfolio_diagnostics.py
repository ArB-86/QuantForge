import pandas as pd

TOP_N = 10

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

# =====================
# BUILD PORTFOLIO PICKS
# =====================

all_picks = []

for date, group in df.groupby("Date"):

    picks = (
        group
        .sort_values(
            "ENSEMBLE_PRED",
            ascending=False
        )
        .head(TOP_N)
        .copy()
    )

    picks["PortfolioDate"] = date

    all_picks.append(
        picks[
            [
                "PortfolioDate",
                "Ticker",
                "Close",
                "ENSEMBLE_PRED",
                "TARGET_5D_RETURN"
            ]
        ]
    )

picks_df = pd.concat(
    all_picks,
    ignore_index=True
)

# =====================
# PRICE ANALYSIS
# =====================

print("\n====================")
print("PRICE ANALYSIS")
print("====================")

print(
    picks_df["Close"]
    .describe()
)

# =====================
# STOCK FREQUENCY
# =====================

print("\n====================")
print("TOP 20 MOST PICKED")
print("====================")

freq = (
    picks_df["Ticker"]
    .value_counts()
    .head(20)
)

print(freq)

# =====================
# UNIQUE STOCKS
# =====================

print("\n====================")
print("UNIQUE STOCKS")
print("====================")

print(
    "Unique Stocks:",
    picks_df["Ticker"]
    .nunique()
)

# =====================
# TURNOVER
# =====================

dates = sorted(
    picks_df["PortfolioDate"]
    .unique()
)

overlaps = []

for i in range(
    1,
    len(dates)
):

    prev_set = set(
        picks_df[
            picks_df["PortfolioDate"]
            == dates[i - 1]
        ]["Ticker"]
    )

    curr_set = set(
        picks_df[
            picks_df["PortfolioDate"]
            == dates[i]
        ]["Ticker"]
    )

    overlap = (
        len(
            prev_set
            &
            curr_set
        )
        /
        TOP_N
    )

    overlaps.append(
        overlap
    )

print("\n====================")
print("TURNOVER ANALYSIS")
print("====================")

print(
    "Average Overlap:",
    round(
        pd.Series(overlaps).mean(),
        4
    )
)

print(
    "Median Overlap:",
    round(
        pd.Series(overlaps).median(),
        4
    )
)

print(
    "Average Turnover:",
    round(
        1 -
        pd.Series(overlaps).mean(),
        4
    )
)

# =====================
# RETURN PROFILE
# =====================

print("\n====================")
print("RETURN PROFILE")
print("====================")

print(
    picks_df[
        "TARGET_5D_RETURN"
    ].describe()
)
