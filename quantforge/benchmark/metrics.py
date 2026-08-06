import numpy as np
import pandas as pd


def compute_benchmark_metrics(
    portfolio,
    benchmark,
    config=None,
    strategy_cagr=None,          # pre‑computed CAGR from backtest
    holding_days=20,             # matches config["holding_days"]
):
    # --- benchmark return column -------------------------------------------------
    if "Return" in benchmark.columns:
        bench_ret_col = "Return"
    elif config and "benchmark_return_column" in config:
        bench_ret_col = config["benchmark_return_column"]
    else:
        raise ValueError("Benchmark return column not found.")
    if bench_ret_col not in benchmark.columns:
        raise ValueError(f"Column '{bench_ret_col}' missing from benchmark.")

    # --- merge on Date -----------------------------------------------------------
    port = portfolio[["Date", "Return"]].copy()
    bench = benchmark[["Date", bench_ret_col]].copy()

    port = port.rename(columns={"Return": "Strategy_Return"})
    bench = bench.rename(columns={bench_ret_col: "Benchmark_Return"})

    merged = pd.merge(port, bench, on="Date", how="inner")
    if len(merged) == 0:
        return {}

    strategy = merged["Strategy_Return"]
    benchmark_returns = merged["Benchmark_Return"]

    # --- canonical CAGR using the backtest convention (years = n * holding_days / 252)
    n = len(strategy)
    years = (n * holding_days) / 252

    # Only compute benchmark CAGR here; strategy CAGR is taken from the backtest
    benchmark_equity = (1 + benchmark_returns).cumprod()
    benchmark_cagr = benchmark_equity.iloc[-1] ** (1 / years) - 1

    # --- portfolio‑relative metrics ----------------------------------------------
    covariance = np.cov(strategy, benchmark_returns)[0, 1]
    benchmark_variance = benchmark_returns.var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else np.nan
    alpha = strategy.mean() - beta * benchmark_returns.mean()

    active = strategy - benchmark_returns
    tracking_error = active.std(ddof=1)
    information_ratio = active.mean() / tracking_error if tracking_error > 0 else np.nan

    correlation = strategy.corr(benchmark_returns)

    # Use the provided strategy CAGR if given, else fall back to canonical calculation
    if strategy_cagr is None:
        strategy_equity = (1 + strategy).cumprod()
        strategy_cagr = strategy_equity.iloc[-1] ** (1 / years) - 1

    return {
        "Benchmark CAGR": float(benchmark_cagr),
        "Strategy CAGR": float(strategy_cagr),
        "Alpha": float(alpha),
        "Beta": float(beta),
        "Correlation": float(correlation),
        "Tracking Error": float(tracking_error),
        "Information Ratio": float(information_ratio),
        "Excess CAGR": float(strategy_cagr - benchmark_cagr),
    }
