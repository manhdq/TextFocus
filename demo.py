import os
import json
import glob
from datetime import datetime
from shutil import rmtree
import cv2
import pickle
import yaml
import numpy as np

import torch
import onnxruntime as rt

from utils import FocusChip
from utils.parser import demo_parser
from utils.misc import remove_prefix
from retinafocus import RetinaFocus, RetinaFocusWrapper
from prediction import TextPrediction

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']


def prepare_configs(cfg_path):
    '''
    Prepare configs to run inference
    '''
    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        demo_cfg = cfg['demo']

    return demo_cfg


def prepare_model(model_path, model_type, device='cpu'):
    """
    Prepare model to run inference
    """
    assert model_type in ['torch', 'onnx'], 'Only support `torch` and `onnx` model types!'

    if model_type == 'torch':
        ##### LOAD CHECKPOINT #####
        print("Loading checkpoint ...")
        states = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_cfg = states['model_cfg']
        data_cfg = states['data_cfg']

        ##### CREATE MODEL #####
        print("Creating model ...")
        model = RetinaFocus(cfg=model_cfg['retinafocus'], phase='test')
        model = model.to(device)
        model.eval()

        states['model'] = remove_prefix(states['model'], 'module.')
        model.load_state_dict(states['model'])
    elif model_type == 'onnx':
        ##### CREATE SESSION #####
        print('Creating session...')
        sess_options = rt.SessionOptions()
        sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        with open(model_path, 'rb') as f:
            states = pickle.load(f)
        onnx_binary = states['model']
        model_cfg = states['model_cfg']
        data_cfg = states['data_cfg']
        model = rt.InferenceSession(onnx_binary, sess_options=sess_options)

    wrapper_model = RetinaFocusWrapper(model)

    return wrapper_model, model_cfg, data_cfg


def start_demo_pipeline(img_root,
                        model_path,
                        cfg_path,
                        save_dir,
                        model_type,
                        device='cpu'):
    """
    Start demo pipeline
    """
    demo_cfg = prepare_configs(cfg_path=cfg_path)

    ##### PREPARING MODEL #####
    model, model_cfg, data_cfg = prepare_model(model_path=model_path, model_type=model_type, device=device)

    ##### CREATE FOCUSCHIP GENERATOR #####
    foc_chip_gen = FocusChip(demo_cfg['focus_threshold'],
                            kernel_size=demo_cfg['kernel_size'],
                            min_chip_size=demo_cfg['min_chip_size'],
                            stride=model_cfg['retinafocus']['autofocus']['stride'])

    ##### PREPARING DATA #####
    if img_root.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
        img_list = [img_root]
    else:
        img_list = glob.glob(os.path.join(img_root, "*.*"))
        img_list = [img_p for img_p in img_list if img_p.split('.').lower() in ALLOWED_EXTENSIONS]

    demo_tool = TextPrediction(model=model,
                            foc_chip_gen=foc_chip_gen,
                            priorbox_cfg=model_cfg['priorbox'],
                            data_cfg=data_cfg,
                            demo_cfg=demo_cfg,
                            preds_save_dir=save_dir)

    demo_results = dict()
    prediction_times = []
    for img_p in img_list:
        img_name = img_p.split(os.sep)[-1]
        img = cv2.imread(img_p)

        # Create save folder for current sample
        cur_save_dir = os.path.join(save_dir, img_name.split('.')[0])
        if os.path.exists(cur_save_dir):
            rmtree(cur_save_dir)
        os.makedirs(cur_save_dir)

        demo_result = demo_tool.predict_an_image(img_name=img_name, img=img)
        demo_results.update(demo_result)
        prediction_times.append(demo_result[img_name]['prediction_time'])
    print(f"FPS: {1. / np.mean(prediction_times):.3f}")
    return demo_results


def main():
    ##### GET CONFIGURATION #####
    args = demo_parser()
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    if os.path.exists(args.save_dir):
        rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    demo_results = start_demo_pipeline(img_root=args.img_root,
                                    model_path=args.model_path,
                                    cfg_path=args.cfg_path,
                                    save_dir=args.save_dir,
                                    model_type=args.model_type,
                                    device=device)
    with open(os.path.join(args.save_dir, '_meta_data.json'), 'w') as f:
        json.dump(demo_results, f, indent=4)
        

if __name__ == '__main__':
    start = datetime.now()
    main()
    print('Total time:', datetime.now() - start)