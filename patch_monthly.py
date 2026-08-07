from pathlib import Path

mp = Path("quantforge/walkforward/monthly.py")
lines = mp.read_text().splitlines()

# Identify the block to replace
start, end = None, None
for i, line in enumerate(lines):
    if "num_gpus = min(self.workers, mp.cpu_count())" in line:
        start = i
    if start is not None and "train_time = time.perf_counter() - overall - fold_gen_time" in line:
        end = i
        break

if start is not None and end is not None:
    new_block = [
        '        print("Windows compatibility: running walkforward sequentially")',
        '',
        '        global _G_FEAT, _G_TARG, _G_DF',
        '        _G_FEAT = self.feature_matrix',
        '        _G_TARG = self.target_vector',
        '        _G_DF = self.df_for_test',
        '',
        '        out_dir = str(self.prediction_file.parent)',
        '',
        '        _worker(0, tasks, out_dir)',
    ]
    lines[start:end] = new_block
    mp.write_text("\n".join(lines))
    print("MonthlyLoop patched for Windows sequential execution.")
else:
    print("Could not locate the exact block. start:", start, "end:", end)
