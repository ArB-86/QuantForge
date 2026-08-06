from abc import ABC, abstractmethod

import pandas as pd


class EnsembleModel(ABC):

    name = "base"

    @abstractmethod
    def predict(
        self,
        predictions: dict[str, pd.Series],
    ) -> pd.Series:
        ...
