from pathlib import Path
be = Path("quantforge/backtest/backtest_engine.py")
lines = be.read_text().splitlines()

# Find the start of the broken block (line with 'holdings = build_portfolio(')
start = None
end = None
for i, line in enumerate(lines):
    if 'holdings = build_portfolio(' in line:
        start = i
    if start is not None and 'top_k=self.config.get("top_n", 10),' in line:
        # This is part of our correct block; find where the broken stuff ends
        # Look for a line that has only ')' with the same indentation
        for j in range(i+1, len(lines)):
            if lines[j].strip() == ')' and lines[j].startswith('        '):
                end = j
                break
        break

if start is not None and end is not None:
    new_block = [
        '        holdings = build_portfolio(',
        '            predictions,',
        '            method=self.config.get("portfolio", "inverse_vol"),',
        '            score_column="PRED_RETURN",',
        '            top_k=self.config.get("top_n", 10),',
        '            buffer_k=self.config.get("buffer_k", 15),',
        '            rebalance_freq=self.config.get("holding_days", 5),',
        '        )',
    ]
    lines[start:end+1] = new_block
    be.write_text('\n'.join(lines))
    print('Backtest engine fixed.')
else:
    print('Could not locate the exact block. start:', start, 'end:', end)
