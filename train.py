# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function
import os
import warnings
import argparse
warnings.filterwarnings("ignore")

from config import Config
from models import get_model, get_loss
from metrics import get_metric
from datasets import get_dataloader
from trainer import Trainer

parser = argparse.ArgumentParser('Training EfficientDet')
parser.add_argument('--config_path', default="./config/configs.yaml")
parser.add_argument('--print_per_iter', type=int, default=10, help='Number of iteration to print')
parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
parser.add_argument('--save_interval', type=int, default=100, help='Number of steps between saving')
parser.add_argument('--resume', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')
parser.add_argument('--saved_path', type=str, default='./weights')
parser.add_argument('--freeze_backbone', action='store_true', help='whether to freeze the backbone')

args = parser.parse_args()

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    trainloader, valloader = get_dataloader(config)

    criterion = get_loss(config.loss, config.using_autofocus).cuda()
    metric = get_metric(config)
    
    model = get_model(config.model, config.using_autofocus)

    trainer = Trainer(args=args,
                      config=config,
                      model=model,
                      criterion=criterion,
                      metric=metric,
                      train_loader=trainloader,
                      val_loader=valloader,
                      using_autofocus=config.using_autofocus)
    trainer.train()


if __name__ == '__main__':
    config = Config(args.config_path)
    main(config)
