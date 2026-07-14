import pandas as pd

from .base import AlphaModel
from .registry import register
from .utils import normalize_cross_section


@register
class LiquidityAlpha(
    AlphaModel
):

    name = "liquidity"

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        score = (
            0.45 * df["LOG_DOLLAR_VOLUME"]
            + 0.30 * df["SIZE_RANK"]
            - 0.25 * df["ILLIQUIDITY"]
        )

        return normalize_cross_section(
            df,
            score,
        )
