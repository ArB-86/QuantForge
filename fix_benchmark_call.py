from pathlib import Path

runner = Path("quantforge/research_pipeline/runner.py")
code = runner.read_text()

old = "benchmark_stats = compute_benchmark_metrics(portfolio, benchmark)"
new = """benchmark_stats = compute_benchmark_metrics(
            portfolio,
            benchmark,
            config=self.config,
            strategy_cagr=metrics.get("CAGR"),
            holding_days=self.config.get("holding_days", 20),
        )"""

if old in code:
    code = code.replace(old, new)
    runner.write_text(code)
    print("Benchmark call updated.")
else:
    print("Old benchmark call not found. Searching for compute_benchmark_metrics...")
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if "compute_benchmark_metrics" in line:
            print(f"Line {i+1}: {line}")
