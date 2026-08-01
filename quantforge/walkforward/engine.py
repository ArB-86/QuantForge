import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from quantforge.walkforward.report import WalkForwardReport
from quantforge.walkforward.splitter import WalkForwardSplitter
from quantforge.walkforward.context import WalkForwardResult
from quantforge.walkforward.leaderboard import WalkForwardLeaderboard
from quantforge.walkforward.statistics import WalkForwardStatistics
from quantforge.walkforward.stability import StabilityAnalyzer


class WalkForwardEngine:
    def __init__(self, runner, config):
        self.config = config
        self.runner_class = runner.__class__

    def run(self, dates, train_size, test_size):
        splitter = WalkForwardSplitter(dates, train_size, test_size)
        report = WalkForwardReport()
        windows = splitter.split()

        results = []
        start_time = time.perf_counter()

        for i, window in enumerate(windows, start=1):
            print("=" * 80)
            print(f"Walk Forward Window {i}/{len(windows)}")
            print(window)
            print("=" * 80)

            cfg = dict(self.config)
            cfg["train_start"] = str(window.train_start)
            cfg["train_end"]   = str(window.train_end)
            cfg["test_start"]  = str(window.test_start)
            cfg["test_end"]    = str(window.test_end)

            window_runner = self.runner_class(cfg)
            context = window_runner.run()

            report.add(window, context.metrics)
            results.append(WalkForwardResult(
                window=window,
                experiment_id=context.experiment_id,
                metrics=context.metrics,
                experiment_dir=context.artifacts["run_dir"],
            ))

        elapsed = time.perf_counter() - start_time
        out_dir = Path("results/walkforward")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Windows report
        df = report.dataframe()
        summary_df = report.summary()
        df.to_csv(out_dir / "walkforward_windows.csv", index=False)
        summary_df.to_csv(out_dir / "walkforward_summary.csv")

        # Advanced statistics
        stats = WalkForwardStatistics.build(df)
        stats.to_csv(out_dir / "walkforward_statistics.csv")

        # Stability
        stability = {}
        for col in ["Sharpe", "CAGR", "Score"]:
            if col in df.columns:
                stability[f"{col} Stability"] = StabilityAnalyzer.score(df[col])

        with open(out_dir / "walkforward_stability.json", "w") as f:
            json.dump(stability, f, indent=4, default=str)

        # Leaderboard
        leaderboard = WalkForwardLeaderboard.build(results)
        leaderboard.to_csv(out_dir / "walkforward_leaderboard.csv", index=False)

        # Run registry
        runs_df = pd.DataFrame({
            "Experiment": [r.experiment_id for r in results],
            "Directory":  [r.experiment_dir for r in results],
        })
        runs_df.to_csv(out_dir / "walkforward_runs.csv", index=False)

        # Summary metrics
        best_score = max(r.metrics.get("Score", float('-inf')) for r in results)
        avg_score = sum(r.metrics.get("Score", 0.0) for r in results) / len(results)

        summary = {
            "windows": len(results),
            "best_score": best_score,
            "average_score": avg_score,
            "completed": True,
            "generated_at": datetime.utcnow().isoformat(),
            "runtime_seconds": round(elapsed, 2),
        }

        with open(out_dir / "walkforward_summary.json", "w") as f:
            json.dump(summary, f, indent=4, default=str)

        # Metrics JSON (same structure as a regular experiment)
        metrics = {
            "best_score": best_score,
            "average_score": avg_score,
            "total_windows": len(results),
            "runtime_seconds": summary["runtime_seconds"],
            "best_window": results[0].experiment_id if results else None,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Markdown report
        with open(out_dir / "report.md", "w") as f:
            f.write(f"# Walk-Forward Report\n\n")
            f.write(f"**Model**: {self.config.get('model', 'lightgbm')}\n")
            f.write(f"**Windows**: {len(results)}\n")
            f.write(f"**Best Score**: {best_score:.4f}\n")
            f.write(f"**Average Score**: {avg_score:.4f}\n")
            f.write(f"**Runtime**: {elapsed:.2f}s\n\n")
            f.write("## Leaderboard\n\n")
            f.write(leaderboard.to_markdown(index=False))

        print(f"\nWalk-forward reports saved to {out_dir}")
        return report, results
