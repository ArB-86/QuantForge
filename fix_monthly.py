from pathlib import Path

mp = Path("quantforge/walkforward/monthly.py")
code = mp.read_text()

old = """num_gpus = min(self.workers, mp.cpu_count())
chunks = [[] for _ in range(num_gpus)]
for idx, task in enumerate(tasks):
    gpu_id = idx % num_gpus
    chunks[gpu_id].append(task)

print(f"Distributing {n_folds} folds over {num_gpus} GPUs")

# Set globals for workers
global _G_FEAT, _G_TARG, _G_DF
_G_FEAT = self.feature_matrix
_G_TARG = self.target_vector
_G_DF   = self.df_for_test

out_dir = str(self.prediction_file.parent)

procs = []
for gpu_id, chunk in enumerate(chunks):
    if not chunk:
        continue
    p = mp.Process(target=_worker, args=(gpu_id, chunk, out_dir))
    p.start()
    procs.append(p)

for p in procs:
    p.join()"""

new = """print("Windows compatibility: running walkforward sequentially")

global _G_FEAT, _G_TARG, _G_DF
_G_FEAT = self.feature_matrix
_G_TARG = self.target_vector
_G_DF = self.df_for_test

out_dir = str(self.prediction_file.parent)

_worker(
    0,
    tasks,
    out_dir,
)"""

if old in code:
    code = code.replace(old, new)
    mp.write_text(code)
    print("MonthlyLoop updated for Windows sequential execution.")
else:
    print("Old block not found. Checking file...")
    print(code[-500:])
