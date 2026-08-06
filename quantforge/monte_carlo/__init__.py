"""Monte Carlo stress-testing engine."""

import numpy as np
import pandas as pd


def monte_carlo_var(returns, confidence=0.95, simulations=10000):
    """Estimate Value at Risk using Monte Carlo simulation."""
    returns = returns.dropna().values
    mean = returns.mean()
    std = returns.std(ddof=1)
    simulated = np.random.normal(mean, std, (simulations, len(returns)))
    final_values = (1 + simulated).prod(axis=1)
    var = np.percentile(final_values, 100 * (1 - confidence))
    return float(var)


def stress_test(returns, scenarios=None):
    """Apply historical or custom stress scenarios to portfolio returns."""
    if scenarios is None:
        scenarios = {"2008_crisis": -0.05, "2020_covid": -0.07, "rate_hike": -0.03}
    results = {}
    for name, shock in scenarios.items():
        stressed = returns + shock
        results[name] = {
            "shock": shock,
            "mean_return": float(stressed.mean()),
            "volatility": float(stressed.std(ddof=1)),
            "var_95": float(np.percentile(stressed, 5)),
        }
    return results


def simulate_drawdown_paths(returns, n_paths=1000):
    """Generate random drawdown paths for worst-case analysis."""
    returns = returns.dropna().values
    path_results = []
    for _ in range(n_paths):
        path = np.random.choice(returns, size=len(returns), replace=True)
        equity = (1 + path).cumprod()
        peak = np.maximum.accumulate(equity)
        drawdown = (equity / peak) - 1
        path_results.append(drawdown.min())
    return {
        "mean_worst_dd": float(np.mean(path_results)),
        "median_worst_dd": float(np.median(path_results)),
        "p95_worst_dd": float(np.percentile(path_results, 5)),
    }
