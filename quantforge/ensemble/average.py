import pandas as pd

from .base import EnsembleModel
from .registry import register


@register
class AverageEnsemble(EnsembleModel):

    name = "average"

    def predict(
        self,
        predictions,
    ):

        df = pd.DataFrame(predictions)

        return df.mean(axis=1)
