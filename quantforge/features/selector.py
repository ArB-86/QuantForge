TARGET_COLUMNS = {
    "TARGET_1D",
    "TARGET_5D_RETURN",
    "TARGET_10D_RETURN",
    "TARGET_20D_RETURN",
    "TARGET_5D_CLASS",
}

EXCLUDE_COLUMNS = {
    "Date",
    "Ticker",
}


def get_training_features(df):

    features = []

    for c in df.columns:

        if c in TARGET_COLUMNS:
            continue

        if c in EXCLUDE_COLUMNS:
            continue

        features.append(c)

    return sorted(features)
