import pandas as pd

from quantforge.ensemble import EnsembleEngine


def compare_models(df):

    prediction_columns = [

        c

        for c in df.columns

        if c.startswith("PRED_")

    ]

    predictions = {

        c: df[c]

        for c in prediction_columns

    }

    methods = [

        "average",

        "rank_average",

        "weighted",

    ]

    rows = []

    for method in methods:

        engine = EnsembleEngine(
            method
        )

        pred = engine.predict(
            predictions
        )

        ic = (

            df.assign(
                ENSEMBLE=pred
            )

            .groupby("Date")

            .apply(

                lambda x:

                x["ENSEMBLE"].corr(

                    x["TARGET_20D_RETURN"],

                    method="spearman",

                )

            )

        )

        rows.append({

            "Method": method,

            "MeanIC": ic.mean(),

            "MedianIC": ic.median(),

            "PositiveIC": (ic > 0).mean(),

        })

    result = pd.DataFrame(rows)

    print()
    print("=" * 80)
    print("ENSEMBLE COMPARISON")
    print("=" * 80)
    print(result)

    result.to_csv(
        "results/ensemble_comparison.csv",
        index=False,
    )
