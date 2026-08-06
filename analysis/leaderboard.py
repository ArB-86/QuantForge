import json
import pandas as pd
from pathlib import Path

def generate_leaderboard():
    history_dir = Path("results/experiments")
    data = []
    
    for folder in history_dir.glob("*"):
        if folder.is_dir():
            metrics_path = folder / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    try:
                        metrics = json.load(f)
                        metrics["ExpID"] = folder.name
                        data.append(metrics)
                    except json.JSONDecodeError:
                        print(f"Skipping corrupt file: {metrics_path}")
    
    if not data:
        print("No metrics found.")
        return

    df = pd.DataFrame(data)
    # Fill missing keys with NaN so code doesn't crash
    cols = ["Sharpe", "CAGR", "MaxDD", "ExpID"]
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
            
    print(df.sort_values(by="Sharpe", ascending=False).to_string())

generate_leaderboard()