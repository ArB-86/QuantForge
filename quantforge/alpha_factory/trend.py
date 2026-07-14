import pandas as pd

from .base import AlphaModel
from .registry import register
from .utils import normalize_cross_section


@register
class TrendAlpha(
    AlphaModel
):

    name = "trend"

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        score = (
            0.40 * df["EMA20_OVER_EMA200"]
            + 0.20 * df["EMA20_DISTANCE"]
            + 0.20 * df["EMA50_DISTANCE"]
            + 0.20 * df["PRICE_VS_VWAP"]
        )

        return normalize_cross_section(
            df,
            score,
        )
