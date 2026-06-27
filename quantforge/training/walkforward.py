import pandas as pd


def monthly_walkforward_split(
    df,
    train_years=5,
    purge_days=5,
    date_column="Date",
):

    dates = sorted(
        pd.to_datetime(
            df[date_column]
        ).unique()
    )

    months = (
        pd.Series(dates)
        .dt.to_period("M")
        .unique()
    )

    for month in months:

        test_start = (
            month.to_timestamp()
        )

        test_end = (
            month.to_timestamp("M")
        )

        train_end = (
            test_start
            - pd.Timedelta(
                days=purge_days + 1
            )
        )

        train_start = (
            train_end
            - pd.DateOffset(
                years=train_years
            )
        )

        train = df[
            (df[date_column] >= train_start)
            &
            (df[date_column] <= train_end)
        ]

        test = df[
            (df[date_column] >= test_start)
            &
            (df[date_column] <= test_end)
        ]

        if len(train) == 0 or len(test) == 0:
            continue

        yield (
            str(month),
            train,
            test,
        )
