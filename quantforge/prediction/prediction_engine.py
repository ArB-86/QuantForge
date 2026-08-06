from pathlib import Path

import joblib
import pandas as pd

from quantforge.ensemble import EnsembleEngine


class PredictionEngine:

    def __init__(
        self,
        config,
    ):

        self.config = config

    def _load_model(
        self,
        path,
    ):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        return joblib.load(path)

    def predict(
        self,
        df,
    ):

        models = self.config["models"]

        predictions = {}

        for name, path in models.items():

            model = self._load_model(path)

            features = model.feature_name_

            predictions[name] = pd.Series(
                model.predict(
                    df[features]
                ),
                index=df.index,
            )

        if len(predictions) == 1:

            return next(
                iter(
                    predictions.values()
                )
            )

        method = self.config.get(
            "ensemble_method",
            "rank_average",
        )

        engine = EnsembleEngine(
            method,
        )

        return engine.predict(
            predictions,
        )
