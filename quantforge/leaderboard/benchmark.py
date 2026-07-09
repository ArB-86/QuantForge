from quantforge.leaderboard.engine_engine import (
    LeaderboardEngine,
)

from quantforge.research.comparison import (
    ExperimentComparison,
)


class LeaderboardComparison:

    def __init__(self):

        self.lb = LeaderboardEngine()

    def compare(
        self,
        experiment_a,
        experiment_b,
    ):

        a = self.lb.get(
            experiment_a,
        )

        b = self.lb.get(
            experiment_b,
        )

        metrics_a = {

            "Sharpe": a["sharpe"],
            "CAGR": a["cagr"],
            "Max Drawdown": a["maxdd"],
            "Win Rate": a["winrate"],

        }

        metrics_b = {

            "Sharpe": b["sharpe"],
            "CAGR": b["cagr"],
            "Max Drawdown": b["maxdd"],
            "Win Rate": b["winrate"],

        }

        return ExperimentComparison(

            metrics_a,

            metrics_b,

        ).compare()