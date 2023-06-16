import os
import argparse
import cv2
import numpy as np
import warnings
import glob
import copy
import pickle
import onnxruntime as rt
from datetime import datetime
from PIL import Image
from shutil import rmtree
warnings.filterwarnings("ignore")

import torch

from cfglib.config import config as cfg, update_config, print_config
from cfglib.option import BaseOptions, DemoOptions
from network.textnet import TextBPNFocus
from network.wrapper import TextNetWrapper
from prediction import TextBPNFocusPrediction
from utils.focus_chip_generator import FocusChip
from utils.augmentation import BaseTransform

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


def get_parser():
    '''
    Parse arguments of the demo
    '''
    parser = argparse.ArgumentParser(description='Demo script.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to TextFocus model.')
    parser.add_argument('--cfg-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to save output images')
    parser.add_argument('--model-type', type=str,
                        help='Type of model [torch, onnx].')
    parser.add_argument('--no-cuda', action='store_true',
                        help='do not using cuda')
    parser.add_argument('--img-root', type=str,
                        help="Image root folder / file for demo")

    args = parser.parse_args()
    return args


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def prepare_model(model_path, model_type, device="cpu"):
    """
    Prepare model to run inference
    """
    assert model_type in ['torch', 'onnx'], 'Only support `torch` and `onnx` model types!'

    if model_type == 'torch':
        ##### CREATE MODEL #####
        print("Creating model ...")
        model = TextBPNFocus(backbone=cfg.net, is_training=False,
                        using_autofocus=cfg.enable_autofocus)
        model = model.to(device)
        model.eval()

        ##### LOAD CHECKPOINT #####
        print("Loading checkpoint ...")
        load_model(model, cfg.model_path)

    elif model_type == 'onnx':
        ##### CREATE SESSION #####
        print('Creating session...')
        sess_options = rt.SessionOptions()
        sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        with open(model_path, 'rb') as f:
            states = pickle.load(f)
        onnx_binary = states['model']
        model = rt.InferenceSession(onnx_binary, sess_options=sess_options)

    wrapper_model = TextNetWrapper(model)

    return wrapper_model


def start_demo_pipeline(img_root,
                        model_path,
                        save_dir,
                        model_type):
    """
    Start demo pipeline
    """

    ##### PREPARE MODEL #####
    model = prepare_model(model_path=cfg.model_path, model_type=cfg.model_type, device=cfg.device)

    ##### PREPARE DATA #####
    if isinstance(cfg.img_root, str):
        if cfg.img_root.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            img_list = [cfg.img_root]
        else:
            img_list = glob.glob(os.path.join(cfg.img_root, "*"))
            img_list = [img_path for img_path in img_list if img_path.split('.')[-1].lower() in ALLOWED_EXTENSIONS]
    else:
        ##TODO:
        raise
    img_list = sorted(img_list)
    
    cfg.test_size = copy.deepcopy(cfg.input_size)
    cfg.test_size = [cfg.test_size] * 2 if isinstance(cfg.test_size, int) else cfg.test_size
    
    transform = BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)

    ##### FOCUSCHIP GENERATOR #####
    foc_chip_gen = FocusChip(cfg.focus_threshold,
                            kernel_size=cfg.kernel_size,
                            min_chip_size=cfg.min_chip_size,
                            stride=cfg.autofocus_stride)
    
    ##TODO: Code TextPrediction without autofocus
    demo_tool = TextBPNFocusPrediction(model=model,
                                    transform=transform,
                                    foc_chip_gen=foc_chip_gen)

    demo_results = dict()
    prediction_times = []
    ##TODO: priority
    for img_p in img_list:
        img_name = img_p.split(os.sep)[-1]

        img = pil_load_img(img_p)
        try:
            h, w, c = img.shape
            assert c == 3
        except:
            img = cv2.imread(img_p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)

        demo_result = demo_tool.predict_an_image(img_name=img_name, img=img)
        demo_results.update(demo_result)
        prediction_times.append(demo_result[img_name]['prediction_time'])

    print(f"FPS: {1. / np.mean(prediction_times):.3f}")
    return demo_results


def main():
    if os.path.exists(args.save_dir):
        rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "txt_preds"))

    demo_results = start_demo_pipeline(img_root=args.img_root,
                                    model_path=args.model_path,
                                    save_dir=args.save_dir,
                                    model_type=args.model_type)


if __name__ == '__main__':
    start = datetime.now()

    # Parse arguments
    option = DemoOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()

    print('Total time:', datetime.now() - start)