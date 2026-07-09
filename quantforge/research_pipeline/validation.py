
class ValidationManager:

    def __init__(self, context):
        self.context = context

    def validate(self):

        required = [
            "checkpoint",
            "model",
            "experiment_dir",
        ]

        missing = [
            k for k in required
            if k not in self.context.artifacts
        ]

        if missing:
            raise RuntimeError(
                "Missing artifacts: " + ", ".join(missing)
            )

        if not self.context.metrics:
            raise RuntimeError(
                "Metrics not generated."
            )

        self.context.status = "VALIDATED"
