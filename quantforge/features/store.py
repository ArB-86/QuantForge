from quantforge.features.momentum import add_momentum_features
from quantforge.features.volatility import add_volatility_features
from quantforge.features.trend import add_trend_features
from quantforge.features.market import add_market_features
from quantforge.features.statistical import add_statistical_features
from quantforge.features.liquidity import add_liquidity_features


class FeatureStore:

    def build(self, df):

        print("Adding momentum features...")
        df = add_momentum_features(df)

        print("Adding volatility features...")
        df = add_volatility_features(df)

        print("Adding trend features...")
        df = add_trend_features(df)

        print("Adding market features...")
        df = add_market_features(df)

        print("Adding statistical features...")
        df = add_statistical_features(df)

        print("Adding liquidity features...")
        df = add_liquidity_features(df)

        print("Feature engineering complete.")

        return df