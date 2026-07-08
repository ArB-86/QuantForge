#!/usr/bin/env bash
set -e

echo "========================================"
echo "QuantForge Repository Checks"
echo "========================================"

echo "[1/6] Python compilation..."
python -m compileall quantforge

echo "[2/6] Git status..."
git status --short

echo "[3/6] Python file count..."
find quantforge -name "*.py" | wc -l

echo "[4/6] TODO count..."
grep -RInE "TODO|FIXME|XXX|HACK" quantforge || true

echo "[5/6] Empty modules..."
find quantforge -name "*.py" -size 0 || true

echo "[6/6] Large modules..."
find quantforge -name "*.py" -exec wc -l {} + | sort -nr | head -20

echo
echo "Repository checks completed successfully."
