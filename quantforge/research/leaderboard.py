from quantforge.storage.database.experiments import ExperimentDB


class Leaderboard:

    def __init__(self):

        self.db = ExperimentDB()

    def top(
        self,
        n=20,
    ):

        return self.db.top(n)

    def champion(self):

        rows = self.db.top(1)

        if len(rows):

            return rows[0]

        return None