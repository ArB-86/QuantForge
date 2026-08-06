from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT.parent / "data"

CHECKPOINTS = DATA / "checkpoints"

TRAINING = DATA / "training"

MODELS = ROOT.parent / "models"

RESULTS = ROOT.parent / "results"


def checkpoint(name):

    return CHECKPOINTS / name


def training(name):

    return TRAINING / name


def model(name):

    return MODELS / name


def result(name):

    return RESULTS / name
