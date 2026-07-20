import json
from pathlib import Path
import sys

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    # find latest experiment
    exps = sorted(Path("results/experiments").glob("EXP_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not exps:
        print("No experiments found.")
        sys.exit(1)
    path = exps[0] / "runtime.json"

if not path.exists():
    print(f"runtime.json not found at {path}")
    sys.exit(1)

with open(path) as f:
    data = json.load(f)

print("=" * 60)
print("RUNTIME DASHBOARD")
print("=" * 60)
for k, v in data.items():
    if isinstance(v, float):
        print(f"{k:30}: {v:10.2f}s")
    else:
        print(f"{k:30}: {v}")
