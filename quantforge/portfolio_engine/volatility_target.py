import numpy as np
import pandas as pd


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
        self.last_leverage = None

    def apply(self, returns):

        # Step 1: Volatility estimation with floor
        vol = (
            returns
            .rolling(
                self.lookback,
                min_periods=self.lookback,
            )
            .std()
            .clip(lower=1e-4)
            * np.sqrt(252)
        )

        # Step 2: Leverage calculation with smoothing
        leverage = (
            self.target_vol / vol
        ).ewm(
            span=5,
            adjust=False,
        ).mean()

        # Clamp leverage
        leverage = leverage.clip(
            self.min_leverage,
            self.max_leverage,
        )

        leverage = leverage.fillna(1.0)

        # ----- DEBUG: print leverage stats -----
        print(
            "Leverage:",
            leverage.min(),
            leverage.mean(),
            leverage.max(),
        )
        # ---------------------------------------

        # Step 3: Apply scaling and store diagnostics
        scaled = returns * leverage

        self.last_leverage = leverage

        return scaled
