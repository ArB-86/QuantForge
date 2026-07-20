import numpy as np
import pandas as pd

from .base import EnsembleModel
from .registry import register


@register
class WeightedEnsemble(EnsembleModel):

    name = "weighted"

    def __init__(
        self,
        weights=None,
    ):

        self.weights = weights

    def predict(
        self,
        predictions,
    ):

        df = pd.DataFrame(predictions)

        if self.weights is None:

            w = np.ones(
                len(df.columns)
            )

        else:

            w = np.asarray(
                self.weights,
                dtype=float,
            )

        w = w / w.sum()

        return (
            df.values @ w
        )
