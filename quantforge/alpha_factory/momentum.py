import pandas as pd

from .base import AlphaModel
from .registry import register
from .utils import normalize_cross_section


@register
class MomentumAlpha(
    AlphaModel
):

    name = "momentum"

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        score = (
            0.30 * df["RETURN_250D"]
            + 0.25 * df["RETURN_120D"]
            + 0.20 * df["RETURN_60D"]
            + 0.15 * df["RETURN_20D"]
            + 0.10 * df["MOM_120_60"]
        )

        return normalize_cross_section(
            df,
            score,
        )
