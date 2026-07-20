REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get_ensemble(name):

    if name not in REGISTRY:
        raise KeyError(name)

    return REGISTRY[name]()


def list_ensembles():

    return sorted(REGISTRY)
