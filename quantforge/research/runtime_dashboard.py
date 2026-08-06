import json
import time
from pathlib import Path

class RuntimeDashboard:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.data = {}
        self.timers = {}

    def start_timer(self, name):
        self.timers[name] = time.perf_counter()

    def stop_timer(self, name):
        if name in self.timers:
            elapsed = time.perf_counter() - self.timers[name]
            self.data[name] = elapsed
            del self.timers[name]
            return elapsed
        return None

    def record(self, key, value):
        self.data[key] = value

    def save(self):
        print("[DASHBOARD] Saving runtime.json...")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "runtime.json", "w") as f:
            json.dump(self.data, f, indent=2)
        print("[DASHBOARD] Saved.")
