from pathlib import Path

configs = sorted(
    Path("configs/generated").glob("*.json")
)

with open(
    "queue/jobs.txt",
    "w"
) as fp:

    for cfg in configs:

        fp.write(
            str(cfg) + "\n"
        )

print(
    len(configs),
    "jobs queued"
)
