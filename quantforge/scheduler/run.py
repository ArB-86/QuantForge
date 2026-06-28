import subprocess
from pathlib import Path

jobs = [
    j.strip()
    for j in Path(
        "queue/jobs.txt"
    ).read_text().splitlines()
    if j.strip()
]

for job in jobs:

    print()

    print("="*80)
    print("RUNNING", job)
    print("="*80)

    subprocess.run(
        [
            "python",
            "experiment.py",
            job
        ],
        check=True
    )
