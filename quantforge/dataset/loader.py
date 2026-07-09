import pandas as pd


def load_training_data(
    dataset_path,
    date_column="Date",
):

    df = pd.read_csv(
        dataset_path
    )

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    return (
        df
        .sort_values(
            [date_column, "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )


def trading_dates(
    df,
    date_column="Date",
):

    return sorted(
        df[date_column]
        .unique()
    )
