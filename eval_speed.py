import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataloader import Ctw1500Text_New
from network.textnet import TextNet
from utils.augmentation import BaseTransform
from utils.misc import to_device, mkdirs
from config_lib.config import config as cfg, update_config, print_config
from config_lib.option import BaseOptions

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def inference(model, test_loader, output_dir):
    total_time = 0.0
    if cfg.exp_name != "MLT2017" and cfg.exp_name != "ArT":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)

    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        input_dict["img"] = to_device(image)
        # init model
        if i == 0:
            output_dict = model(input_dict, test_speed=True)

        for k in range(0, 50):
            start = time.time()
            output_dict = model(input_dict, test_speed=True)
            torch.cuda.synchronize()
            end = time.time()
            total_time += end - start

        fps = (i + 1) * 50 / total_time

        print(
            "detect {} / {} images: {}. ({:.2f} fps)".format(
                i + 1, len(test_loader), meta["image_id"][0], fps
            )
        )


def main(vis_dir_path):
    osmkdir(vis_dir_path)

    testset = Ctw1500Text_New(
        data_root="data/CTW1500",
        is_training=False,
        transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
    )

    if cfg.cuda:
        cudnn.benchmark = True

    test_loader = data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(
        cfg.save_dir,
        cfg.exp_name,
        "TextBPN_{}_{}.pth".format(model.backbone_name, cfg.checkepoch),
    )

    model.load_model(model_path)
    model = model.to(cfg.device)  # copy to cuda
    model.eval()
    with torch.no_grad():
        print("Start testing TextBPN++.")
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        inference(model, test_loader, output_dir)


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, "{}_test".format(cfg.exp_name))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)
