from pathlib import Path
import subprocess

jobs = Path(
    "queue/jobs.txt"
).read_text().splitlines()

for job in jobs:

    job = job.strip()

    if not job:
        continue

    print("="*80)
    print(job)
    print("="*80)

    subprocess.run(
        [
            "python",
            "experiment.py",
            job
        ],
        check=True
    )
