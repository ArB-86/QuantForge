import pandas as pd

from .base import AlphaModel
from .registry import register
from .utils import normalize_cross_section


@register
class MeanReversionAlpha(
    AlphaModel
):

    name = "mean_reversion"

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        score = (
            -0.40 * df["ZSCORE_20D"]
            -0.30 * df["RSI14"]
            -0.30 * df["RETURN_5D"]
        )

        return normalize_cross_section(
            df,
            score,
        )
