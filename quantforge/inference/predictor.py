from __future__ import annotations
import json, joblib
from pathlib import Path
import pandas as pd
from quantforge.dataset.builder import DatasetBuilder

class Predictor:
    def __init__(self,experiment_dir):
        self.experiment_dir=Path(experiment_dir)
        with open(self.experiment_dir/"config.json","r") as f: self.config=json.load(f)
        self.model=joblib.load(self.experiment_dir/"model.pkl")
        self.builder=DatasetBuilder(data_path=self.config["data_path"],features=self.config["features"],target=self.config["target"])
        self.features=self.builder.features
    def load_dataset(self): return self.builder.prepare()
    def predict(self):
        df=self.load_dataset()
        latest_date=df["Date"].max(); df=df[df["Date"]==latest_date].copy()
        missing=[c for c in self.features if c not in df.columns]
        if missing: raise RuntimeError(f"Missing features: {missing}")
        X=df[self.features].astype("float32")
        if X.shape[1]!=len(self.features): raise RuntimeError(f"Expected {len(self.features)} features, got {X.shape[1]}")
        pred=self.model.predict(X)
        result=df[["Date","Ticker"]].copy(); result["PRED_RETURN"]=pred
        return result
