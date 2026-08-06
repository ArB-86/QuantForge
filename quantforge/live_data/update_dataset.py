from __future__ import annotations
from pathlib import Path
import pandas as pd

class DatasetUpdater:
    def __init__(self,dataset_path): self.dataset_path=Path(dataset_path)
    def update(self,new_data:pd.DataFrame):
        print("="*80); print("UPDATING MASTER DATASET"); print("="*80)
        master=pd.read_parquet(self.dataset_path)
        last_date=master["Date"].max(); print("Last date in master:",last_date.date())
        new_data=new_data[new_data["Date"]>last_date].copy(); print("Rows after date filter:",len(new_data))
        if new_data.empty: print("Dataset already up-to-date."); return master
        master["Date"]=pd.to_datetime(master["Date"]); new_data["Date"]=pd.to_datetime(new_data["Date"])
        combined=pd.concat([master,new_data],ignore_index=True)
        combined=combined.drop_duplicates(subset=["Date","Ticker"],keep="last")
        combined=combined.sort_values(["Date","Ticker"])
        combined.to_parquet(self.dataset_path,index=False,compression="zstd")
        print("Updated rows:",len(combined))
        return combined
