from pathlib import Path

be = Path("quantforge/backtest/backtest_engine.py")
code = be.read_text()

# Insert column creation right before "holdings = build_portfolio("
old = "holdings = build_portfolio("
new = """    # Ensure required columns exist for the allocator
    if "TARGET_5D" not in predictions.columns:
        predictions["TARGET_5D"] = predictions.get("TARGET_20D_RETURN", predictions.get("RETURN_5D", 0))
    if "RET_1D" not in predictions.columns:
        predictions["RET_1D"] = predictions.get("RETURN_1D", 0)
    if "Raw_Prediction" not in predictions.columns:
        predictions["Raw_Prediction"] = predictions["PRED_RETURN"]
    if "Prediction" not in predictions.columns:
        predictions["Prediction"] = predictions["PRED_RETURN"]

    holdings = build_portfolio("""

code = code.replace(old, new)
be.write_text(code)
print("Added missing columns before build_portfolio call.")
