from pathlib import Path
be = Path("quantforge/backtest/backtest_engine.py")
code = be.read_text()

# The old block that we need to replace (it contains top_n)
old_block = '''holdings = build_portfolio(

    predictions,

    method=self.config.get(
        "portfolio",
        "equal_weight",
    ),

    score_column="PRED_RETURN",

    top_n=self.config["top_n"],

    max_stock_weight=self.config.get(
        "max_stock_weight",
        1.0,
    ),

)'''

# The correct call using the actual function signature
new_block = '''holdings = build_portfolio(
    predictions,
    method=self.config.get("portfolio", "inverse_vol"),
    score_column="PRED_RETURN",
    top_k=self.config.get("top_n", 10),
    buffer_k=self.config.get("buffer_k", 15),
    rebalance_freq=self.config.get("holding_days", 5),
)'''

if old_block in code:
    code = code.replace(old_block, new_block)
    be.write_text(code)
    print("Backtest engine call fixed.")
else:
    # Fallback: directly replace any line that contains 'top_n=self.config'
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if 'top_n=self.config' in line:
            lines[i] = '        top_k=self.config.get("top_n", 10),'
        if 'max_stock_weight' in line:
            lines[i] = '        buffer_k=self.config.get("buffer_k", 15),'
        if 'holding_days' not in line and 'rebalance_freq' not in line:
            pass
    # Also replace the method if still "equal_weight"
    for i, line in enumerate(lines):
        if '"equal_weight"' in line and 'method' in line:
            lines[i] = '        method=self.config.get("portfolio", "inverse_vol"),'
    code = "\n".join(lines)
    be.write_text(code)
    print("Backtest engine call fixed (fallback method).")
