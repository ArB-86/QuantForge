import pandas as pd

from .momentum import MomentumAlpha
from .trend import TrendAlpha
from .volatility import VolatilityAlpha
from .liquidity import LiquidityAlpha
from .mean_reversion import MeanReversionAlpha
from .utils import normalize_cross_section


class CompositeAlpha:

    def __init__(
        self,
        prediction_column="PRED_RETURN",
    ):

        self.prediction_column = prediction_column

        self.momentum = MomentumAlpha()
        self.trend = TrendAlpha()
        self.volatility = VolatilityAlpha()
        self.liquidity = LiquidityAlpha()
        self.mean_reversion = MeanReversionAlpha()

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        prediction = normalize_cross_section(
            df,
            df[self.prediction_column],
        )

        momentum = self.momentum.compute(df)

        trend = self.trend.compute(df)

        volatility = self.volatility.compute(df)

        liquidity = self.liquidity.compute(df)

        mean_reversion = self.mean_reversion.compute(df)

        composite = (
            0.35 * prediction
            + 0.25 * momentum
            + 0.15 * trend
            + 0.10 * volatility
            + 0.10 * liquidity
            + 0.05 * mean_reversion
        )

        return normalize_cross_section(
            df,
            composite,
        )
