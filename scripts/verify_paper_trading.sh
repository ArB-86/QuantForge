#!/usr/bin/env bash
set -e

echo "=================================================="
echo " QuantForge Paper Trading Verification"
echo "=================================================="

python -m compileall quantforge

echo
echo "[1/6] Smoke"
python -m quantforge.paper_trading.tests.test_smoke

echo
echo "[2/6] Orders"
python -m quantforge.paper_trading.tests.test_orders

echo
echo "[3/6] Portfolio"
python -m quantforge.paper_trading.tests.test_portfolio

echo
echo "[4/6] Engine"
python -m quantforge.paper_trading.tests.test_engine

echo
echo "[5/6] Bridge"
python -m quantforge.paper_trading.tests.test_bridge

echo
echo "[6/6] CLI"
python -m quantforge.cli.paper

echo
echo "=================================================="
echo " PAPER TRADING VERIFIED"
echo "=================================================="
