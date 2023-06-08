import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataloader.text_data import TextData
from network.textnet import TextNet
from utils.augmentation import BaseTransform
from config_lib.config import config as cfg, update_config, print_config
from config_lib.option import BaseOptions
from utils.misc import to_device, mkdirs

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
    if cfg.exp_name == "Totaltext":
        testset = TextData(
            data_root="data/total-text-mat",
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
        )

    elif cfg.exp_name == "Ctw1500":
        testset = TextData(
            data_root="data/ctw1500",
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
        )
    elif cfg.exp_name == "Icdar2015":
        testset = TextData(
            data_root="data/Icdar2015",
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
        )
    elif cfg.exp_name == "MLT2017":
        testset = TextData(
            data_root="data/MLT2017",
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
        )
    elif cfg.exp_name == "TD500":
        testset = TextData(
            data_root="data/TD500",
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
        )
    elif cfg.exp_name == "ArT":
        testset = TextData(
            data_root="data/ArT",
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
        )
    else:
        print("{} is not justify".format(cfg.exp_name))

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
        "{}_{}.pth".format(model.backbone_name, cfg.checkepoch),
    )

    model.load_model(model_path)
    model = model.to(cfg.device)
    model.eval()
    with torch.no_grad():
        print("Start testing model")
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        inference(model, test_loader, output_dir)


if __name__ == "__main__":
    # Parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, "{}_test".format(cfg.exp_name))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)

    main(vis_dir)
