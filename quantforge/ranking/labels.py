import pandas as pd

def make_rank_labels(

    df,

    target="TARGET_20D_RETURN"

):

    out = df.copy()

    out["LABEL"] = (

        out
        .groupby("Date")[target]
        .rank(
            ascending=False,
            method="first"
        )

    )

    return out
