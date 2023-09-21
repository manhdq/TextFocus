import warnings
warnings.filterwarnings("ignore")

import os
import time
import cv2
import numpy as np
from typing import List

import torch
import torch.nn.functional as F
from mmcv import Config

from pydantic import BaseModel
from fastapi import FastAPI, status

from . import utils
from models import PAN_FOCUS_PREDICT
from models.utils import fuse_module


app = FastAPI()


class TextFocusInput(BaseModel):
    img_base64: str


class TextFocusOutput(BaseModel):
    bboxes: List
    scores: List[float]
    time: float
    num_tile: int


def convert_to_list(bboxes):
    list_out = []
    for bbox in bboxes:
        list_out.append(bbox.tolist())
    return list_out


@app.post("/text_focus_detect", status_code=status.HTTP_202_ACCEPTED)
def text_focus_detect(item_input: TextFocusInput):
    img_base64 = item_input.img_base64

    img_pil = utils.base64_to_image(img_base64)
    img = np.array(img_pil).astype(np.uint8)
    
    with torch.no_grad():
        outputs = model(img)

    # Each outputs bbox is a numpy array, convert it to list
    outputs["bboxes"] = convert_to_list(outputs["bboxes"])
    # Each outputs score is numpy value, convert it to float
    outputs["scores"] = [float(score) for score in outputs["scores"]]

    # Delete 'scores', 'time', 'num_tile'
    del outputs["scores"], outputs["time"], outputs["num_tile"]

    return outputs


# Initalize model
config_path = "config/pan/pan_r18_ctw_chip_demo.py"
cfg = Config.fromfile(config_path)
cfg.update(dict(vis=False))  ##TODO: Delete later

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
    raise

# fuse conv and bn
model = fuse_module(model)