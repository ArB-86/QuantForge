import subprocess
import time

STEPS = [

    ("5D Model",
     "python backtesting/monthly_walkforward_regression_v12.py"),

    ("10D Model",
     "python backtesting/monthly_walkforward_regression_v13.py"),

    ("20D Model",
     "python backtesting/monthly_walkforward_regression_v14.py"),

    ("Ensemble",
     "python backtesting/ensemble_builder_v1.py"),

    ("Model Comparison",
     "python -m experiments.model_comparison"),

    ("Transaction Cost Sweep",
     "python -m experiments.transaction_cost_sweep"),

    ("Top-N Sweep",
     "python -m experiments.topn_sweep")

]

start = time.time()

for name, cmd in STEPS:

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    ret = subprocess.call(cmd, shell=True)

    if ret != 0:
        raise RuntimeError(f"FAILED: {name}")

print()
print("=" * 60)
print("QUANTFORGE COMPLETE")
print("=" * 60)
print(f"Elapsed {(time.time()-start)/60:.2f} minutes")
