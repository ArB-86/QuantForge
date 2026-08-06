REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get_alpha(name):

    if name not in REGISTRY:
        raise KeyError(
            f"Unknown alpha: {name}"
        )

    return REGISTRY[name]()


def list_alphas():
    return sorted(REGISTRY.keys())
