import subprocess
from concurrent.futures import ThreadPoolExecutor

JOBS = [

    ("GPU1",1),

    ("GPU2",2),

    ("GPU3",3),

    ("GPU4",4),

    ("GPU5",5),

    ("GPU6",6),

    ("GPU7",7)

]

SCRIPT = "backtesting/lightgbm_optuna_search.py"


def worker(job):

    name,gpu = job

    print(f"Starting {name}")

    subprocess.run(

        [
            "python",
            SCRIPT
        ],

        env={
            **__import__("os").environ,
            "CUDA_VISIBLE_DEVICES":str(gpu)
        }

    )

with ThreadPoolExecutor(
    max_workers=7
) as pool:

    pool.map(worker,JOBS)

print()

print("ALL OPTUNA JOBS FINISHED")
