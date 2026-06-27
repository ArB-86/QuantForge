import subprocess
from concurrent.futures import ThreadPoolExecutor

JOBS = [

("configs/lgbm_5d.json",1),

("configs/lgbm_10d.json",2),

("configs/lgbm_20d.json",3)

]

def run(job):

    config,gpu=job

    print()

    print(config)

    subprocess.run(

        [
            "python",
            "train_lgbm.py",
            config
        ],

        env={
            **__import__("os").environ,
            "CUDA_VISIBLE_DEVICES":str(gpu)
        }

    )

with ThreadPoolExecutor(max_workers=3) as ex:

    ex.map(run,JOBS)

print()

print("ALL FINISHED")
