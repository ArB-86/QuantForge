from __future__ import annotations

from quantforge.storage.database.experiments import ExperimentDB


class ExperimentLogger:
    def __init__(self):
        self.db = ExperimentDB()

    def log(self, config, metrics, score, study=None):
        row = {
            "name": config["name"],
            "model": config["model"],
            "Sharpe": metrics["Sharpe"],
            "CAGR": metrics["CAGR"],
            "Max Drawdown": metrics["Max Drawdown"],
            "Win Rate": metrics["Win Rate"],
            "Score": score,
            "params": str(config),
        }

        if study is not None:
            row["study_name"] = getattr(study, "study_name", None)
            row["trial_count"] = len(getattr(study, "trials", []))
            row["best_trial_number"] = getattr(getattr(study, "best_trial", None), "number", None)
            row["best_params"] = str(getattr(study, "best_params", None))
            row["best_value"] = getattr(study, "best_value", None)

        self.db.insert(row)
