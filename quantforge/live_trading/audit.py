from dataclasses import asdict
from pathlib import Path
import json


class AuditLogger:

    def __init__(self, output_dir="results/live_trading"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.file = self.output_dir / "audit.jsonl"

    def log(self, fill):

        if hasattr(fill, "__dataclass_fields__"):
            record = asdict(fill)
        else:
            record = dict(fill)

        with self.file.open("a") as f:
            f.write(json.dumps(record) + "\n")
