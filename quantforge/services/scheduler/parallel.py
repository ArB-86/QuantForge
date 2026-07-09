import subprocess
from pathlib import Path

jobs = [
    j.strip()
    for j in Path(
        "queue/jobs.txt"
    ).read_text().splitlines()
    if j.strip()
]

processes = []

for job in jobs:

    print("Launching", job)

    p = subprocess.Popen(
        [
            "python",
            "experiment.py",
            job
        ]
    )

    processes.append(p)

for p in processes:
    p.wait()

print()
print("ALL EXPERIMENTS FINISHED")
