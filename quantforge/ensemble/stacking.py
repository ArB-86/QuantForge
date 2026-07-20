import pandas as pd
from sklearn.linear_model import Ridge

from .base import EnsembleModel
from .registry import register


@register
class StackingEnsemble(EnsembleModel):

    name = "stacking"

    def __init__(self):

        self.model = Ridge(
            alpha=1.0,
        )

        self.fitted = False

    def fit(
        self,
        predictions,
        target,
    ):

        X = pd.DataFrame(
            predictions
        )

        self.model.fit(
            X,
            target,
        )

        self.fitted = True

    def predict(
        self,
        predictions,
    ):

        X = pd.DataFrame(
            predictions
        )

        if not self.fitted:

            raise RuntimeError(
                "Stacking model not fitted."
            )

        return pd.Series(
            self.model.predict(X),
            index=X.index,
        )
