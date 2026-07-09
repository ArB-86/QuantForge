from pathlib import Path
import shutil


class ArtifactManager:

    def __init__(self, experiment_dir):

        self.root = Path(experiment_dir)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_json(self, name, src):

        shutil.copy2(
            src,
            self.root / name,
        )

    def save_csv(self, name, src):

        shutil.copy2(
            src,
            self.root / name,
        )

    def save_parquet(self, name, src):

        shutil.copy2(
            src,
            self.root / name,
        )

    def save_model(self, src):

        shutil.copy2(
            src,
            self.root / "model.txt",
        )

    def path(self):

        return self.root
