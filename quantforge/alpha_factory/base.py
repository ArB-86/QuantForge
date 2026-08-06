from abc import ABC, abstractmethod

import pandas as pd


class AlphaModel(ABC):

    name = "base"

    @abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Return a cross-sectional alpha score.
        """
        raise NotImplementedError
