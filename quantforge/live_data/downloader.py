from __future__ import annotations
import os,time
os.environ["YFINANCE_USE_CACHE"]="0"
import yfinance as yf
import pandas as pd

class MarketDataDownloader:
    def download(self,tickers,period="5d",batch_size=30):
        print("="*80); print("DOWNLOADING MARKET DATA"); print("="*80)
        frames=[]; total_batches=(len(tickers)+batch_size-1)//batch_size
        for i in range(0,len(tickers),batch_size):
            batch=tickers[i:i+batch_size]; batch_no=i//batch_size+1
            print(f"Batch {batch_no}/{total_batches} ({len(batch)} symbols)")
            success=False
            for attempt in range(3):
                try:
                    df=yf.download(tickers=batch,period=period,auto_adjust=False,progress=False,threads=False,group_by="ticker")
                    if len(df): frames.append(df)
                    success=True; break
                except Exception as e: print(f"Retry {attempt+1}/3 failed:",e); time.sleep(2)
            if not success: print("Skipping batch:",batch_no)
            time.sleep(1)
        if not frames: raise RuntimeError("No data downloaded.")
        return pd.concat(frames,axis=1)
