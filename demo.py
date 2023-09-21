import argparse
import json
import os
import time
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from mmcv import Config

from models import PAN_FOCUS_PREDICT
from models.utils import fuse_module
from utils import ResultFormat


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print(img_path)
        raise
    return img


def predict(img_path, model, cfg):
    model.eval()

    if not os.path.exists(img_path):
        raise

    img = get_img(img_path)
    img_name = img_path.split(os.sep)[-1].split('.')[0]

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    start = time.time()
    # forward
    with torch.no_grad():
        outputs = model(img)

        print('output', outputs)
    end = time.time()


def main(args):
    cfg = Config.fromfile(args.config)
    cfg.update(dict(vis=args.vis))

    # model
    model = PAN_FOCUS_PREDICT(cfg=cfg)
    model = model.cuda()

    if os.path.isfile(cfg.checkpoint):
        print("Loading model and optimizer from checkpoint '{}'".format(
            cfg.checkpoint))

        checkpoint = torch.load(cfg.checkpoint)
        model_state_dict = checkpoint['state_dict']
        new_model_state_dict = model.state_dict()

        leftover_state_names = []
        for key, _ in new_model_state_dict.items():
            if "module." + key in model_state_dict:
                new_model_state_dict[key] = model_state_dict["module." + key]
            else:
                leftover_state_names.append(key)
                
        model.load_state_dict(new_model_state_dict)
        print("State names not exists in loaded checkpoint:")
        for state_name in leftover_state_names:
            print(f"- {state_name}")
    else:
        print("No checkpoint found at '{}'".format(args.resume))
        raise

    # fuse conv and bn
    model = fuse_module(model)
    # model_structure(model)

    # predict
    predict(args.img_path, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', default='config/pan/pan_r18_ctw_finetune.py', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    main(args)