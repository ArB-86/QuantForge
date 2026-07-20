import pandas as pd


def rolling_ic(df, freq="M"):

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    rows = []

    for period, x in df.groupby(df["Date"].dt.to_period(freq)):

        ic = x["PRED_RETURN"].corr(
            x["TARGET_20D_RETURN"],
            method="spearman",
        )

        rows.append(
            {
                "Period": str(period),
                "IC": ic,
                "N": len(x),
            }
        )

    out = pd.DataFrame(rows)

    print("=" * 80)
    print("ROLLING IC")
    print("=" * 80)
    print(out)

    out.to_csv(
        "results/rolling_ic.csv",
        index=False,
    )

    return out
