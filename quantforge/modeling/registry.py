MODEL_REGISTRY = {}


def register_model(name):

    def decorator(model_cls):
        MODEL_REGISTRY[name] = model_cls
        return model_cls

    return decorator


def get_model(name):

    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}") from exc


def list_models():

    return sorted(MODEL_REGISTRY)