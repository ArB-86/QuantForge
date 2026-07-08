#!/usr/bin/env bash
set -e

echo "========================================"
echo "QuantForge Repository Checks"
echo "========================================"

echo "[1/8] Python compilation..."
python -m compileall quantforge

echo "[2/8] Ruff..."
ruff check quantforge

echo "[3/8] Black..."
black --check quantforge

echo "[4/8] Git status..."
git status --short

echo "[5/8] Python file count..."
find quantforge -name "*.py" | wc -l

echo "[6/8] TODO count..."
grep -RInE "TODO|FIXME|XXX|HACK" quantforge || true

echo "[7/8] Empty modules..."
find quantforge -name "*.py" -size 0 || true

echo "[8/8] Large modules..."
find quantforge -name "*.py" -exec wc -l {} + | sort -nr | head -20

echo
echo "Repository checks completed successfully."
