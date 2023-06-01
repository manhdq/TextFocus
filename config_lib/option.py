import argparse
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime("%b%d_%H-%M-%S") + "\n"

    for k, v in sorted(args_dict.items()):
        option_str += "{}: {}\n".format(str(k), str(v))

    return option_str


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Basic opts
        # self.parser.add_argument(
        #     "--exp_name",
        #     default="CTW1500",
        #     type=str,
        #     choices=[
        #         "CTW1500",
        #         "ALL",
        #     ],
        #     help="Experiment name",
        # )
        self.parser.add_argument("--gpu", default="0", help="Set gpu id", type=str)
        self.parser.add_argument(
            "--resume", default=None, type=str, help="Path to target resume checkpoint"
        )
        self.parser.add_argument(
            "--num_workers",
            default=24,
            type=int,
            help="Number of workers used in dataloading",
        )
        self.parser.add_argument(
            "--cuda", default=True, type=str2bool, help="Use cuda to train model"
        )
        self.parser.add_argument(
            "--mgpu", action="store_true", help="Use multi-gpu to train model"
        )
        self.parser.add_argument(
            "--save_dir", default="./model/", help="Path to save checkpoint models"
        )
        self.parser.add_argument(
            "--vis_dir", default="./vis/", help="Path to save visualization images"
        )
        self.parser.add_argument(
            "--log_dir", default="./logs/", help="Path to tensorboard log"
        )
        self.parser.add_argument(
            "--loss", default="CrossEntropyLoss", type=str, help="Training Loss"
        )
        # self.parser.add_argument('--input_channel', default=1, type=int, help='number of input channels' )
        self.parser.add_argument(
            "--pretrain",
            default=False,
            type=str2bool,
            help="Pretrained AutoEncoder model",
        )
        self.parser.add_argument(
            "--verbose",
            "-v",
            default=True,
            type=str2bool,
            help="Whether to output debug info",
        )
        self.parser.add_argument(
            "--viz", action="store_true", help="Whether to output debug info"
        )
        # self.parser.add_argument('--viz', default=True, type=str2bool, help='Whether to output debug info')

        # Train opts
        self.parser.add_argument(
            "--max_epoch", default=250, type=int, help="Max epochs"
        )
        self.parser.add_argument(
            "--lr",
            "--learning-rate",
            default=1e-3,
            type=float,
            help="Initial learning rate",
        )
        self.parser.add_argument(
            "--lr_adjust",
            default="fix",
            choices=["fix", "poly"],
            type=str,
            help="Learning Rate Adjust Strategy",
        )
        self.parser.add_argument(
            "--stepvalues",
            default=[],
            nargs="+",
            type=int,
            help="# Of iter to change lr",
        )
        self.parser.add_argument(
            "--weight_decay",
            "--wd",
            default=0.0,
            type=float,
            help="Weight decay for SGD",
        )
        self.parser.add_argument(
            "--gamma", default=0.1, type=float, help="Gamma update for SGD lr"
        )
        self.parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
        self.parser.add_argument(
            "--batch_size", default=6, type=int, help="Batch size for training"
        )
        self.parser.add_argument(
            "--optim",
            default="Adam",
            type=str,
            choices=["SGD", "Adam"],
            help="Optimizer",
        )
        self.parser.add_argument(
            "--save_freq", default=5, type=int, help="Save weights every # epoch"
        )
        self.parser.add_argument(
            "--display_freq",
            default=10,
            type=int,
            help="Display training metrics every # iter",
        )
        self.parser.add_argument(
            "--viz_freq",
            default=50,
            type=int,
            help="Visualize training process every # iter",
        )
        self.parser.add_argument(
            "--log_freq",
            default=10000,
            type=int,
            help="Log to tensorboard every # iterations",
        )
        self.parser.add_argument(
            "--val_freq",
            default=1000,
            type=int,
            help="Do validation every # iterations",
        )

        # Backbone
        self.parser.add_argument(
            "--scale", default=1, type=int, help="Prediction on 1/scale feature map"
        )
        self.parser.add_argument(
            "--net",
            default="resnet50",
            type=str,
            choices=[
                "vgg",
                "resnet18",
                "resnet50",
                "deformable_resnet18",
                "deformable_resnet50",
            ],
            help="Network architecture",
        )
        # Data args
        self.parser.add_argument(
            "--load_memory", default=False, type=str2bool, help="Load data into memory"
        )
        self.parser.add_argument(
            "--rescale", type=float, default=255.0, help="Rescale factor"
        )
        self.parser.add_argument(
            "--input_size", default=640, type=int, help="Model input size"
        )
        self.parser.add_argument(
            "--test_size", default=[640, 960], type=int, nargs="+", help="Test size"
        )

        # Eval args00
        self.parser.add_argument(
            "--checkepoch", default=1070, type=int, help="Load checkpoint number"
        )
        self.parser.add_argument(
            "--start_epoch", default=0, type=int, help="Start epoch number"
        )
        self.parser.add_argument(
            "--cls_threshold", default=0.875, type=float, help="Threshold of pse"
        )
        self.parser.add_argument(
            "--dis_threshold",
            default=0.35,
            type=float,
            help="Filter the socre < score_i",
        )

        # Demo args
        self.parser.add_argument(
            "--img_root", default=None, type=str, help="Path to deploy images"
        )

    def parse(self, fixed=None):
        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):
        # Parse options
        self.args = self.parse(fixed)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):
        for k, v in extra_options.items():
            setattr(args, k, v)
