#!/usr/bin/env bash
set -e

echo "=================================================="
echo " QuantForge Live Trading Verification"
echo "=================================================="

python -m compileall quantforge >/dev/null

echo
echo "[1/4] Engine"
python -m quantforge.live_trading.tests.test_engine

echo
echo "[2/4] CLI (paper broker)"
python -m quantforge.cli.live --broker paper

echo
echo "[3/4] Main CLI"
python -m quantforge.cli.main live

echo
echo "[4/4] Imports"
python - <<PY
import pkgutil
import quantforge.live_trading as lt

mods = sorted([m.name for m in pkgutil.iter_modules(lt.__path__)])
for m in mods:
    __import__(f"quantforge.live_trading.{m}")
print("PASS")
PY

echo
echo "=================================================="
echo " LIVE TRADING VERIFIED"
echo "=================================================="
