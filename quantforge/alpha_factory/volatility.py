import pandas as pd

from .base import AlphaModel
from .registry import register
from .utils import normalize_cross_section


@register
class VolatilityAlpha(
    AlphaModel
):

    name = "volatility"

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        score = -(
            0.35 * df["VOL_20D"]
            + 0.25 * df["ATR_PCT"]
            + 0.20 * df["BB_WIDTH"]
            + 0.20 * df["PARKINSON_VOL"]
        )

        return normalize_cross_section(
            df,
            score,
        )
