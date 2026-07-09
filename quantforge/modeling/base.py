from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def predict_proba(self, X):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        ...

    @classmethod
    @abstractmethod
    def load(cls, path):
        ...
