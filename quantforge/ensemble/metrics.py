import pandas as pd


def correlation_table(
    predictions,
):

    return (
        pd.DataFrame(
            predictions
        )
        .corr(
            method="spearman"
        )
    )
