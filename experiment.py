import os
import sys
from multiprocessing import Process

from quantforge.experiment.manager import ExperimentManager
from quantforge.engine.trainer import train
from quantforge.engine.backtest import backtest
from quantforge.shap.engine import shap
from quantforge.prediction.engine import prediction
from quantforge.engine.topn import topn

manager = ExperimentManager(sys.argv[1])

exp_id, folder = manager.create()

os.environ["QF_EXPERIMENT_DIR"] = str(folder)

print()
print("="*80)
print("Experiment:", exp_id)
print(folder)
print("="*80)

train()

jobs = [

    Process(target=backtest),

    Process(target=shap),

    Process(target=prediction),

    Process(target=topn),

]

for j in jobs:
    j.start()

for j in jobs:
    j.join()

print()
print("="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
