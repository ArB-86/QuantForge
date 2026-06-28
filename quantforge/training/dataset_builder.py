class DatasetBuilder:

    def __init__(
        self,
        features,
        target
    ):

        self.features = features
        self.target = target

    def build(
        self,
        df
    ):

        X = df[
            self.features
        ]

        y = df[
            self.target
        ]

        return X, y
