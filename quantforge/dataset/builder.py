import time
from pathlib import Path
import pandas as pd

from quantforge.features.registry import get_features
from quantforge.features.store import FeatureStore

_DATASET_CACHE = {}


class DatasetBuilder:
    """
    Loads and prepares the training dataset.
    """

    def __init__(
        self,
        data_path,
        features,
        target,
        drop_features=None,
        drop_prefixes=None,
        keep_prefixes=None,
    ):
        self.data_path = Path(data_path)

        if isinstance(features, str):
            self.features = get_features(features)
        else:
            self.features = features

        self._apply_feature_filters(
            drop_features=drop_features,
            drop_prefixes=drop_prefixes,
            keep_prefixes=keep_prefixes,
        )

        self.target = target

    def _apply_feature_filters(self, drop_features=None, drop_prefixes=None, keep_prefixes=None):
        if not drop_features and not drop_prefixes and not keep_prefixes:
            return
        filtered = set(self.features)
        if drop_features:
            filtered -= set(drop_features)
        if drop_prefixes:
            for f in list(filtered):
                if any(f.startswith(p) for p in drop_prefixes):
                    filtered.remove(f)
        if keep_prefixes:
            new_filtered = set()
            for f in filtered:
                if any(f.startswith(p) for p in keep_prefixes):
                    new_filtered.add(f)
            filtered = new_filtered
        self.features = sorted(filtered)

    def load(self):
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        t0 = time.perf_counter()

        if self.data_path.suffix == ".parquet":
            df = pd.read_parquet(self.data_path)
        else:
            df = pd.read_csv(self.data_path, low_memory=False)

        print(f"Dataset loaded in {time.perf_counter()-t0:.2f} seconds")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "Ticker"], kind="mergesort").reset_index(drop=True)

        # Memory optimization: convert Ticker to categorical
        df["Ticker"] = df["Ticker"].astype("category")

        return df

    def prepare(self):
        key = str(self.data_path)
        if key in _DATASET_CACHE:
            print("=" * 80)
            print("USING CACHED DATASET")
            print("=" * 80)
            return _DATASET_CACHE[key].copy()

        df = self.load()

        print("=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)

        df = FeatureStore().load_or_build(df)

        # ---- Bulk dtype downcasting ----
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols):
            df[float_cols] = df[float_cols].astype('float32')
        int_cols = df.select_dtypes(include=['int64']).columns
        if len(int_cols):
            df[int_cols] = df[int_cols].astype('int32')

        df = df.dropna(subset=self.features + [self.target])

        for c in self.features:
            df[c] = df[c].astype("float32")
        df[self.target] = df[self.target].astype("float32")

        print()
        print("Rows :", len(df))
        print("Cols :", len(df.columns))
        print("Features :", len(self.features))
        print("Feature list:", self.features[:10], "...")

        _DATASET_CACHE[key] = df.copy()
        return df
