import numpy as np


class VolatilityTarget:

    def __init__(
        self,
        target_vol=0.20,
        lookback=20,
        min_leverage=0.5,
        max_leverage=1.5,
    ):
        self.target_vol = target_vol
        self.lookback = lookback
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage

    def apply(self, returns):

        vol = (
            returns
            .rolling(self.lookback)
            .std()
            * np.sqrt(252)
        )

        leverage = (
            self.target_vol / vol
        )

        leverage = leverage.clip(
            self.min_leverage,
            self.max_leverage,
        )

        leverage = leverage.fillna(1.0)

        return returns * leverage
