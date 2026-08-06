from __future__ import annotations
import argparse,subprocess,sys,time
from datetime import datetime

def run_pipeline():
    print("="*80); print("DAILY PIPELINE"); print("="*80)
    print("Started :",datetime.now())
    t0=time.perf_counter()
    subprocess.run([sys.executable,"-m","quantforge.paper_trading.run"],check=True)
    elapsed=time.perf_counter()-t0
    print("="*80); print("PIPELINE FINISHED"); print("="*80)
    print(f"Runtime : {elapsed:.2f} sec"); print("Finished:",datetime.now())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-now",action="store_true")
    args=parser.parse_args()
    if args.run_now: run_pipeline(); return
    print("Daily scheduler started.")
    while True:
        now=datetime.now()
        if now.weekday()<5 and now.hour==18 and now.minute==0:
            run_pipeline(); time.sleep(60)
        time.sleep(1)

if __name__=="__main__": main()
