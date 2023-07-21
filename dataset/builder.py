import dataset


def build_dataset(cfg, focus_gen=None, split="train"):
    root_param = dict()
    for key in cfg:
        if key in ["batch_size", "train", "test"]:
            continue
        root_param[key] = cfg[key]

    data_cfg = cfg.train if split=="train" else cfg.test
    param = dict()
    for key in data_cfg:
        if key == 'type':
            continue
        param[key] = data_cfg[key]

    param.update(root_param)

    data_load = dataset.__dict__[data_cfg.type](focus_gen=focus_gen, **param)
    
    return data_load
