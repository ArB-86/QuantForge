import json
import sys
import subprocess

if len(sys.argv) != 2:
    print("Usage:")
    print("python train_lgbm.py configs/lgbm_5d.json")
    sys.exit(1)

cfg = json.load(open(sys.argv[1]))

target = cfg["target"]
checkpoint = cfg["checkpoint"]
model = cfg["model"]

cmd = [
    "python",
    "backtesting/monthly_walkforward_regression_v15.py"
]

print("=" * 60)
print(cfg["name"])
print("=" * 60)

subprocess.run(cmd)
