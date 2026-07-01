class ChampionManager:
    """
    Keeps track of the current best experiment.
    """

    def __init__(self):

        self.best_score = float("-inf")
        self.best_metrics = None
        self.best_config = None

    def update(
        self,
        config,
        metrics,
    ):

        score = metrics["Score"]

        if score > self.best_score:

            self.best_score = score
            self.best_metrics = metrics
            self.best_config = config

            return True

        return False

    def summary(self):

        return {

            "Score": self.best_score,

            "Metrics": self.best_metrics,

            "Config": self.best_config,

        }
