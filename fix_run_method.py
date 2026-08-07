from pathlib import Path

be = Path("quantforge/backtest/backtest_engine.py")
content = be.read_text()

# We'll replace the entire run() method with a corrected version.
# Find the start of "def run(self):" and the next method definition (or class end).
lines = content.splitlines()
start = None
end = None
for i, line in enumerate(lines):
    if line.strip().startswith("def run(self):"):
        start = i
    if start is not None and i > start and line.strip().startswith("def ") and not line.strip().startswith("def run("):
        end = i
        break
if start and not end:
    end = len(lines)

new_run = [
    '    def run(self):',
    '',
    '        if "predictions_df" in self.config:',
    '            predictions = self.config["predictions_df"].copy()',
    '        else:',
    '            predictions = self.load_predictions()',
    '',
    '        if "PRED_RETURN" not in predictions.columns:',
    '            raise ValueError(',
    '                "Prediction file does not contain PRED_RETURN. "',
    '                "Run walk-forward training first."',
    '            )',
    '',
    '        # Ensure required columns exist for the allocator',
    '        if "TARGET_5D" not in predictions.columns:',
    '            predictions["TARGET_5D"] = predictions.get("TARGET_20D_RETURN", predictions.get("RETURN_5D", 0))',
    '        if "RET_1D" not in predictions.columns:',
    '            predictions["RET_1D"] = predictions.get("RETURN_1D", 0)',
    '        if "Raw_Prediction" not in predictions.columns:',
    '            predictions["Raw_Prediction"] = predictions["PRED_RETURN"]',
    '        if "Prediction" not in predictions.columns:',
    '            predictions["Prediction"] = predictions["PRED_RETURN"]',
    '',
    '        holdings = build_portfolio(',
    '            predictions,',
    '            method=self.config.get("portfolio", "inverse_vol"),',
    '            score_column="PRED_RETURN",',
    '            top_k=self.config.get("top_n", 10),',
    '            buffer_k=self.config.get("buffer_k", 15),',
    '            rebalance_freq=self.config.get("holding_days", 5),',
    '        )',
    '',
    '        print("=" * 80)',
    '        print("Portfolio Method:", self.config["portfolio"])',
    '        print(holdings[["Date", "Ticker", "Weight"]].head(20))',
    '        print("=" * 80)',
    '',
    '        portfolio = simulate(',
    '            holdings,',
    '            return_column=self.config["target"],',
    '            holding_days=self.config["holding_days"],',
    '            round_trip_cost=self.config["transaction_cost"],',
    '        )',
    '',
    '        portfolio["Return"] = VolatilityTarget(',
    '            target_vol=self.config.get("target_volatility", 0.20),',
    '        ).apply(portfolio["Return"])',
    '',
    '        portfolio["Equity"] = (1 + portfolio["Return"]).cumprod()',
    '',
    '        metrics = evaluate(',
    '            portfolio,',
    '            holding_days=self.config["holding_days"],',
    '        )',
    '',
    '        print()',
    '        for k, v in metrics.items():',
    '            print(f"{k} = {v}")',
    '        print()',
    '',
    '        return holdings, portfolio, metrics',
]

lines[start:end] = new_run
be.write_text("\n".join(lines))
print("Backtest engine run() method rewritten with correct indentation.")
