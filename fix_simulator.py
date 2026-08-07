from pathlib import Path

be = Path("quantforge/backtest/backtest_engine.py")
code = be.read_text()

# Add TARGET_20D_RETURN if missing, right before the simulate call
old = "        portfolio = simulate("
new = """        # Ensure the target return column exists for the simulator
        if "TARGET_20D_RETURN" not in holdings.columns:
            if "RETURN_20D" in holdings.columns:
                holdings["TARGET_20D_RETURN"] = holdings["RETURN_20D"]
            elif "RETURN_5D" in holdings.columns:
                holdings["TARGET_20D_RETURN"] = holdings["RETURN_5D"]
            else:
                holdings["TARGET_20D_RETURN"] = 0.0

        portfolio = simulate("""

code = code.replace(old, new)
be.write_text(code)
print("Added TARGET_20D_RETURN fallback before simulate.")
