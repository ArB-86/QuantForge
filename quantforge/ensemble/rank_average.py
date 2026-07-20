import pandas as pd

from .base import EnsembleModel
from .registry import register


@register
class RankAverageEnsemble(EnsembleModel):

    name = "rank_average"

    def predict(
        self,
        predictions,
    ):

        df = pd.DataFrame(predictions)

        ranked = (
            df.rank(
                pct=True,
                axis=0,
            )
        )

        return ranked.mean(axis=1)
