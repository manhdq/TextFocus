import models


def build_model(cfg):
    model_cfg = cfg.model
    param = dict()
    for key in model_cfg:
        if key == 'type':
            continue
        param[key] = model_cfg[key]
    param["using_autofocus"] = cfg.using_autofocus

    model = models.__dict__[model_cfg.type](**param)

    return model
