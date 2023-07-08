# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import pathlib
from . import dataset 


def get_datalist(data_root, data_subroot):
    data_list = []
    img_list = glob.glob(os.path.join(data_root, "Images", data_subroot, "*"))
    
    for img_path in img_list:
        label_path = img_path.replace("/Images/", "/gt/").replace(".jpg", ".txt")

        img_path = pathlib.Path(img_path.strip(' '))
        label_path = pathlib.Path(label_path.strip(' '))
        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
            data_list.append((str(img_path), str(label_path)))

    return data_list


def get_dataloader(config):

    data_root = config.data_root
    train_subroot = config.train_subroot
    train_data_list = get_datalist(data_root, train_subroot)
    val_subroot = config.val_subroot
    val_data_list = get_datalist(data_root, val_subroot)

    trainset = dataset.ImageDataset(
        data_list=train_data_list,
        input_size=config.image_size,
        img_channel=3,
        shrink_ratio=0.5,
        train=True,
        transform=transforms.ToTensor()
    )

    valset = dataset.ImageDataset(
        data_list=val_data_list,
        input_size=736,  ##TODO: priority. Dynamic this
        img_channel=3,
        shrink_ratio=1,
        train=False,
        transform=transforms.ToTensor()
    )

    # trainset = torch.utils.data.Subset(trainset, list(range(30)))
    # valset = torch.utils.data.Subset(valset, list(range(30)))

    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True)

    valloader = DataLoader(
        dataset=valset, 
        batch_size=config.batch_size*2, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True)

    trainloader.dataset_len = len(trainset)
    valloader.dataset_len = len(valset)

    return trainloader, valloader
