import pandas as pd

from .registry import get_ensemble


class EnsembleEngine:

    def __init__(
        self,
        method="average",
    ):

        self.model = get_ensemble(
            method
        )

    def predict(
        self,
        predictions,
    ):

        return self.model.predict(
            predictions
        )
