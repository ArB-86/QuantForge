import json
from pathlib import Path
import pandas as pd


class ResearchReport:

    def __init__(self, results_dir="results/experiments"):
        self.results_dir = Path(results_dir)
        self.experiments = []

    def load_experiments(self):
        """Load all experiment JSON files from the results directory."""
        self.experiments = []
        for f in sorted(self.results_dir.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
                exp_name = f.stem
                metrics = data.get("metrics", {})
                config = data.get("config", {})
                self.experiments.append({
                    "name": exp_name,
                    "metrics": metrics,
                    "config": config,
                })
        return self.experiments

    def build_summary(self):
        """Build a summary DataFrame with all key metrics."""
        rows = []
        for exp in self.experiments:
            m = exp["metrics"]
            rows.append({
                "Experiment": exp["name"],
                "Final Equity": m.get("Final Equity", 0.0),
                "CAGR": m.get("CAGR", 0.0),
                "Sharpe": m.get("Sharpe", 0.0),
                "Sortino": m.get("Sortino", 0.0),
                "Calmar": m.get("Calmar", 0.0),
                "Max Drawdown": m.get("Max Drawdown", 0.0),
                "Volatility": m.get("Volatility", 0.0),
                "Win Rate": m.get("Win Rate", 0.0),
                "Turnover": m.get("Average Turnover", 0.0),
                "Transaction Cost": m.get("Total Transaction Cost", 0.0),
                "Rebalances": m.get("Rebalances", 0),
                "Score": m.get("Score", 0.0),
            })
        df = pd.DataFrame(rows)
        # Sort by Score descending, then Sharpe descending
        df = df.sort_values(["Score", "Sharpe"], ascending=[False, False])
        return df

    def build_leaderboard(self):
        """Alias for build_summary."""
        return self.build_summary()

    def save_csv(self, output_dir="results/research_report"):
        """Save the summary as CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.build_summary()
        df.to_csv(output_dir / "leaderboard.csv", index=False)
        print(f"CSV saved to {output_dir / 'leaderboard.csv'}")

    def save_html(self, output_dir="results/research_report"):
        """Generate an HTML table from the summary."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.build_summary()

        # Format columns for display
        fmt = {
            "Final Equity": "{:.2f}",
            "CAGR": "{:.1%}",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "Calmar": "{:.2f}",
            "Max Drawdown": "{:.1%}",
            "Volatility": "{:.1%}",
            "Win Rate": "{:.1%}",
            "Turnover": "{:.2f}",
            "Transaction Cost": "{:.4f}",
            "Rebalances": "{:.0f}",
            "Score": "{:.3f}",
        }

        # Build HTML table
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                th { background-color: #4CAF50; color: white; text-align: center; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                tr:hover { background-color: #ddd; }
                .best { background-color: #d4edda; }
                .worst { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <h1>Research Report</h1>
            <table>
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Final Equity</th>
                        <th>CAGR</th>
                        <th>Sharpe</th>
                        <th>Sortino</th>
                        <th>Calmar</th>
                        <th>Max DD</th>
                        <th>Volatility</th>
                        <th>Win Rate</th>
                        <th>Turnover</th>
                        <th>Cost</th>
                        <th>Rebalances</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
        """

        for _, row in df.iterrows():
            html += "<tr>"
            html += f"<td><strong>{row['Experiment']}</strong></td>"
            for col in df.columns[1:]:  # skip Experiment column
                val = row[col]
                if col in fmt:
                    html += f"<td>{fmt[col].format(val)}</td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>\n"

        html += """
                </tbody>
            </table>
        </body>
        </html>
        """

        out_file = output_dir / "report.html"
        with open(out_file, "w") as f:
            f.write(html)
        print(f"HTML report saved to {out_file}")

    def run(self, output_dir="results/research_report"):
        """Run the full report generation."""
        self.load_experiments()
        if not self.experiments:
            print("No experiments found.")
            return
        self.save_csv(output_dir)
        self.save_html(output_dir)
        print("\nReport generation complete.")
        print(f"Output directory: {output_dir}")
