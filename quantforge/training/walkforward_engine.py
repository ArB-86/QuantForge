from quantforge.training.trainer import Trainer


class WalkForwardEngine:

    def __init__(self, config):

        self.trainer = Trainer(config)

    def fit(
        self,
        X_train,
        y_train,
        **kwargs
    ):

        self.trainer.fit(
            X_train,
            y_train,
            **kwargs
        )

    def predict(
        self,
        X_test
    ):

        return self.trainer.predict(
            X_test
        )

    def save(
        self,
        path
    ):

        self.trainer.save(path)
