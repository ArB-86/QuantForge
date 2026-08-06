from __future__ import annotations

import pandas as pd


class WalkForwardLeaderboard:

    @staticmethod
    def build(results):

        rows = []

        for r in results:

            m = r.metrics

            rows.append(
                {
                    "Experiment": r.experiment_id,
                    "Score": m.get("Score"),
                    "Sharpe": m.get("Sharpe"),
                    "CAGR": m.get("CAGR"),
                    "MaxDrawdown": m.get("MaxDrawdown"),
                    "Directory": r.experiment_dir,
                }
            )

        df = pd.DataFrame(rows)

        if "Score" in df.columns:
            df = df.sort_values(
                "Score",
                ascending=False,
                ignore_index=True,
            )

        return df
