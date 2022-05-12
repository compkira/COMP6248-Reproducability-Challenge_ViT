import inspect


def bind(func, config: dict):
    required_params = inspect.signature(func).parameters.keys()
    keys = tuple(config.keys())

    for k in keys:
        if k not in required_params:
            config.pop(k)

    return config
