from pathlib import Path

be = Path("quantforge/backtest/backtest_engine.py")
code = be.read_text()

old = '''holdings = build_portfolio(

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

new = '''holdings = build_portfolio(
    predictions,
    method=self.config.get("portfolio", "inverse_vol"),
    score_column="PRED_RETURN",
    top_k=self.config.get("top_n", 10),
    buffer_k=self.config.get("buffer_k", 15),
    rebalance_freq=self.config.get("holding_days", 5),
)'''

if old in code:
    code = code.replace(old, new)
    be.write_text(code)
    print("Backtest engine updated.")
else:
    print("Old block not found. Check the file manually.")
