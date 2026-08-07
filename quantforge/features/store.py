from quantforge.features.base import add_base_features
from pathlib import Path
import pandas as pd

from quantforge.features.momentum import add_momentum_features
from quantforge.features.volatility import add_volatility_features
from quantforge.features.trend import add_trend_features
from quantforge.features.market import add_market_features
from quantforge.features.statistical import add_statistical_features
from quantforge.features.liquidity import add_liquidity_features

CACHE_PATH = Path("data/cache/features_v1.parquet")


class FeatureStore:

    def build(self, df):
        print("Adding base features...")
        df = add_base_features(df)
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

        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols):
            df[float_cols] = df[float_cols].astype("float32")
        print("Feature engineering complete.")
        return df

    def load_or_build(self, df):
        if CACHE_PATH.exists():
            print("Loading feature cache...")
            return pd.read_parquet(CACHE_PATH)

        print("Building feature cache...")
        df = self.build(df)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH, compression="zstd", index=False)
        print(f"Cache saved to {CACHE_PATH}")
        return df
