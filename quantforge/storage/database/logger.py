from quantforge.storage.database.experiments import (
    ExperimentDB
)


class ExperimentLogger:

    def __init__(self):

        self.db = ExperimentDB()

    def log(
        self,
        config,
        metrics,
        score,
    ):

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

        self.db.insert(row)
