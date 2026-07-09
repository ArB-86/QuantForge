import pandas as pd

def build_groups(df):

    groups = (
        df
        .groupby("Date")
        .size()
        .tolist()
    )

    return groups
