from typing import Dict


class FeatureImportanceReport:

    def __init__(self, model):

        self.model = model

    def importance(self) -> Dict[str, float]:

        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not expose feature_importances_")

        names = getattr(self.model, "feature_name_", None)

        if names is None:
            names = [
                f"feature_{i}"
                for i in range(len(self.model.feature_importances_))
            ]

        return dict(
            zip(
                names,
                self.model.feature_importances_,
            )
        )
