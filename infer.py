import os
import json
from shutil import rmtree
from datetime import datetime
import pickle

import yaml
import torch
import onnxruntime as rt

from utils import FocusChip
from utils.misc import remove_prefix
from utils.parser import infer_parser
from retinafocus import RetinaFocus, RetinaFocusWrapper
from data import InferenceDataloader, FocusGenerator
from inference import CountTextInference


def prepare_configs(cfg_path):
    '''
    Prepare configs to run inference
    '''
    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        infer_cfg = cfg['infer']

    return infer_cfg


def prepare_data(data_cfg, image_dir, test_json_path, stride):
    """
    Prepare data to run inference
    """
    print("Preparing data ...")
    focus_gen = FocusGenerator(dont_care_low=data_cfg['dont_care_low'],
                               dont_care_high=data_cfg['dont_care_high'],
                               small_threshold=data_cfg['small_threshold'],
                               stride=stride)
    infer_dataloader = InferenceDataloader(test_folder_path=image_dir,
                                        focus_gen=focus_gen,
                                        gt_path=test_json_path)
    
    return infer_dataloader


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


def start_infer_pipeline(image_dir,
                         model_path,
                         test_json_path,
                         cfg_path,
                         save_dir,
                         early_stop,
                         model_type,
                         device='cpu'):
    '''
    Start inference pipeline
    '''
    infer_cfg = prepare_configs(cfg_path=cfg_path)

    ##### PREPARING MODEL #####
    model, model_cfg, data_cfg = prepare_model(model_path=model_path, model_type=model_type, device=device)

    ##### CREATE FOCUSCHIP GENERATOR #####
    print('Creating focus-chip generator...')
    foc_chip_gen = FocusChip(infer_cfg['focus_threshold'],
                            kernel_size=infer_cfg['kernel_size'],
                            min_chip_size=infer_cfg['min_chip_size'],
                            stride=model_cfg['retinafocus']['autofocus']['stride'])

    ##### PREPARING DATA #####
    infer_dataloader = prepare_data(data_cfg=data_cfg,
                                    image_dir=image_dir,
                                    test_json_path=test_json_path,
                                    stride=model_cfg['retinafocus']['autofocus']['stride'])

    ##### START INFERENCE #####
    print('Predicting')
    infer_tool = CountTextInference(model=model,
                                    foc_chip_gen=foc_chip_gen,
                                    priorbox_cfg=model_cfg['priorbox'],
                                    data_cfg=data_cfg,
                                    infer_cfg=infer_cfg,
                                    preds_save_dir=save_dir,
                                    early_stop=early_stop)
    infer_results = infer_tool.infer(infer_dataloader)

    return infer_results


def main():
    ##### GET CONFIGURATION #####
    args = infer_parser()
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    if os.path.exists(args.save_dir):
        rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    infer_results = start_infer_pipeline(image_dir=args.image_dir,
                                         model_path=args.model_path,
                                         test_json_path=args.test_json_path,
                                         cfg_path=args.cfg_path,
                                         save_dir=args.save_dir,
                                         early_stop=args.early_stop,
                                         model_type=args.model_type,
                                         device=device)

    with open(os.path.join(args.save_dir, '_meta_data.json'), 'w') as f:
        json.dump(infer_results, f, indent=4)


if __name__ == '__main__':
    start = datetime.now()
    main()
    print('Total time:', datetime.now() - start)