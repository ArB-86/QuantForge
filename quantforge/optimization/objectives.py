def objective_sharpe(metrics):
    return metrics["Sharpe"]

def objective_cagr(metrics):
    return metrics["CAGR"]

def objective_calmar(metrics):
    return metrics["CAGR"] / abs(metrics["MaxDD"])
