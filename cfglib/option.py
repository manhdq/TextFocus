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
        self.parser.add_argument(
            "--exp_name",
            default="CTW1500",
            type=str,
            choices=[
                "CTW1500",
                "ALL",
            ],
            help="Experiment name",
        )
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
            "--save_dir", default="./ckpts/", help="Path to save checkpoint models"
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
        self.parser.add_argument(
            "--data_root", type=str,
            help="root diretory for dataset"
        )
        self.parser.add_argument(
            "--train_subroot", type=str, default="train",
            help="root for train dataset part"
        )
        self.parser.add_argument(
            "--val_subroot", type=str, default="val",
            help="root for val dataset part"
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
                "resnet50",
                "resnet18",
                "deformable_resnet18",
                "deformable_resnet50",
            ],
            help="Network architecture",
        )
        # Autofocus
        self.parser.add_argument(
            "--enable_autofocus",
            action="store_true",
            help="Enable AutoFocus for training",
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


        # Autofocus params
        self.parser.add_argument("--alpha", default=1.0, type=float,
                                help="weight for classification loss")
        self.parser.add_argument("--beta", default=3.0, type=float,
                                help="weight for distance loss")
        self.parser.add_argument("--theta", default=.5, type=float,
                                help="weight for direction loss")
        self.parser.add_argument("--gama", default=.05, type=float,
                                help="weight for poly matching loss")
        self.parser.add_argument("--foc_weight", default=1.0, type=float,
                                help="weight for focus loss")

        self.parser.add_argument("--autofocus_dont_care_low", default=3, type=float,
                                help="ignored lower bound for autofocus mask")
        self.parser.add_argument("--autofocus_dont_care_high", default=200, type=float,
                                help="ignored upper bound for autofocus mask")
        self.parser.add_argument("--autofocus_small_threshold", default=50, type=float,
                                help="small threshold for autofocus mask")
        self.parser.add_argument("--autofocus_stride", default=4, type=int,
                                help="stride for autofocus mask")

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


class DemoOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Basic opts
        self.parser.add_argument(
            "--exp_name",
            default="CTW1500",
            type=str,
            help="Experiment name",
        )
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
                "resnet50",
                "resnet18",
                "deformable_resnet18",
                "deformable_resnet50",
            ],
            help="Network architecture",
        )

        # Autofocus
        self.parser.add_argument(
            "--enable_autofocus",
            action="store_true",
            help="Enable AutoFocus for training",
        )
        self.parser.add_argument("--scale_down", default=10, type=int)
        self.parser.add_argument("--first_row_zoom_in", default=5, type=int)
        self.parser.add_argument("--zoom_in_scale", default=3, type=int)
        self.parser.add_argument("--second_round_size_threshold", default=500, type=int)
        self.parser.add_argument("--valid_range", default=[2800, 4200], type=int, nargs=2,
                help="[min_size, max_size] of an image right before forwarding through the model")
        self.parser.add_argument("--max_chip_size", default=320, type=int)
        self.parser.add_argument("--interpolation", default=1, type=int,
                help="0: INTER_NEAREST"
                    "1: INTER_LINEAR"
                    "2: INTER_CUBIC"
                    "3: INTER_AREA"
                    "4: INTER_LANCZOS4")
        self.parser.add_argument("--top_k_before_nms", default=5000, type=int)
        self.parser.add_argument("--top_k_after_nms", default=750, type=int)
        self.parser.add_argument("--nms_threshold", default=0.3, type=float)
        # Gridgenerator
        self.parser.add_argument("--max_valid_size", default=20000, type=int,
                help="The maximum size of the longest axis")
        self.parser.add_argument("--grid_threshold", default=10000, type=int,
                help="Crop an image into grid if the maximum size is bigger than this threshold (for improving accuracy purpose)")
        self.parser.add_argument("--overlap_ratio", default=0.1, type=float,
                help="The overlap ratio between the overlap area and the whole image")
        # Focus branch
        self.parser.add_argument("--focus_threshold", default=0.2, type=float)
        self.parser.add_argument("--kernel_size", default=7, type=int)
        self.parser.add_argument("--min_chip_size", default=50, type=int)
        self.parser.add_argument("--max_focus_rank", default=2, type=int,
                help="The number of times that use auto-focus prediction to zoom in")

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

        # Demo args
        self.parser.add_argument(
            "--img_root", default=None, type=str, help="Path to deploy images"
        )
        self.parser.add_argument(
            "--save_dir", default="./results", type=str,
            help="Path to save demo output images",
        )
        self.parser.add_argument(
            "--model_type", type=str, default="torch", 
            help="Type of model [torch, onnx]"
        )
        self.parser.add_argument(
            "--model_path", type=str,
            help="Trained model weight path"
        )
        self.parser.add_argument(
            "--vis_threshold", default=0.5, type=float, 
        )
        self.parser.add_argument(
            "--cls_threshold", default=0.5, type=float, 
        )
        self.parser.add_argument(
            "--pixel_scalar", default=50, type=int  ##TODO: change name
        )
        # Draw image
        self.parser.add_argument(
            "--draw_valid_range", default=[1024, 2048], type=int, nargs=2,
        )
        self.parser.add_argument(
            "--draw_preds", action="store_true", help="Draw the predictions"
        )
        self.parser.add_argument(
            "--draw_points", action="store_true", help="Draw the landmark keypoints"
        )

        # Autofocus params
        self.parser.add_argument("--alpha", default=1.0, type=float,
                                help="weight for classification loss")
        self.parser.add_argument("--beta", default=3.0, type=float,
                                help="weight for distance loss")
        self.parser.add_argument("--theta", default=.5, type=float,
                                help="weight for direction loss")
        self.parser.add_argument("--gama", default=.05, type=float,
                                help="weight for poly matching loss")
        self.parser.add_argument("--foc_weight", default=1.0, type=float,
                                help="weight for focus loss")

        self.parser.add_argument("--autofocus_dont_care_low", default=3, type=float,
                                help="ignored lower bound for autofocus mask")
        self.parser.add_argument("--autofocus_dont_care_high", default=200, type=float,
                                help="ignored upper bound for autofocus mask")
        self.parser.add_argument("--autofocus_small_threshold", default=50, type=float,
                                help="small threshold for autofocus mask")
        self.parser.add_argument("--autofocus_stride", default=4, type=int,
                                help="stride for autofocus mask")

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