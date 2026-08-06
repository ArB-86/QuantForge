import pandas as pd


def ic_by_regime(df):
    print("=" * 80)
    print("IC BY REGIME")
    print("=" * 80)

    regimes = [
        "BULL_REGIME",
        "HIGH_VOL_REGIME",
    ]

    for regime in regimes:

        if regime not in df.columns:
            continue

        print()
        print(regime)

        for value in sorted(df[regime].dropna().unique()):

            x = df[df[regime] == value]

            daily_ic = (
                x.groupby("Date")
                 .apply(
                    lambda g: g["PRED_RETURN"].corr(
                        g["TARGET_20D_RETURN"],
                        method="spearman",
                    )
                 )
                 .dropna()
            )

            print(
                value,
                "IC =",
                round(daily_ic.mean(), 4),
                "Days =",
                len(daily_ic),
            )
