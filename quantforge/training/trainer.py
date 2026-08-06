from quantforge.modeling.factory import build

class Trainer:

    def __init__(self, config):

        self.model = build(config)

    def fit(self, X, y, **kwargs):

        if "group" in kwargs:

            self.model.fit(
                X,
                y,
                kwargs["group"]
            )

        else:

            self.model.fit(
                X,
                y
            )

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        self.model.save(path)
