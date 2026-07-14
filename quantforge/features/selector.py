from quantforge.features.registry import get_features


def get_training_features(
    df,
    feature_store="v3",
):

    allowed = set(
        get_features(feature_store)
    )

    features = [
        c
        for c in df.columns
        if c in allowed
    ]

    return sorted(features)
