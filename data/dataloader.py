'''
FocusRetina Dataset
'''
import os
import json
import math
import random
from glob import glob

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from .mixup_utils import combine_mixup_imgs_labels, prepare_mixup_params


class FocusRetinaDataset(Dataset):
    '''
    FocusRetina Dataset
    '''

    def __init__(self,
                 json_path,
                 root_path,
                 aug,
                 focus_gen,
                 phase='train',
                 data_cfg=None,
                 mixup_cfg=None,
                 train_bbox_iof_threshold=1,
                 train_min_num_landmarks=3):
        self.root_path = root_path
        self.augmentation = aug
        self.focus_gen = focus_gen
        self.phase = phase
        self.mixup_cfg = mixup_cfg if self.phase == 'train' else None
        self.train_bbox_iof_threshold = train_bbox_iof_threshold
        self.train_min_num_landmarks = train_min_num_landmarks

        if isinstance(json_path, str):
            with open(json_path) as file:
                img_list = json.load(file)
        elif isinstance(json_path, (list, tuple)):
            img_list = []
            for path in json_path:
                with open(path) as file:
                    img_list.extend(json.load(file))

        use_label_list = np.arange(data_cfg['num_classes']).tolist()
        self.chip_list, self.bbox_list, self.label_list, self.lm_list = [], [], [], []
        bbox_ann_keys = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        for img_data in img_list:
            for chip_data in img_data['image_chips']:
                chip_bbox_list, chip_label_list, chip_lm_list = [], [], []
                # Prepare bbox, lm and class for each chip
                for bbox in chip_data['chip_valid_bboxes']:
                    # Check bbox label we want to use
                    if bbox['class'] in use_label_list:
                        chip_bbox_list.append([float(bbox[k]) for k in bbox_ann_keys])
                        ##TODO: We will need landmark later
                        if 'lm' in bbox:
                            chip_lm_list.append([float(bbox['lm'][i][j])
                                                for i, _ in enumerate(bbox['lm'])
                                                for j, _ in enumerate(bbox['lm'][i])])
                        else:
                            chip_lm_list.append([-1.] * 10)
                        chip_label_list.append(bbox['class'])

                self.chip_list.append(chip_data['chip_name'])
                self.bbox_list.append(np.array(chip_bbox_list))
                self.label_list.append(np.expand_dims(np.array(chip_label_list), axis=-1))
                self.lm_list.append(np.array(chip_lm_list))

    def __len__(self):
        return len(self.chip_list)

    def __getitem__(self, index):
        ##TODO: Understand after
        if self.mixup_cfg is not None:
            mix_type, mix_2_pos, ratios = prepare_mixup_params(self.mixup_cfg)
            indexes = random.choices(range(len(self.chip_list)), k=mix_type-1)
            indexes.append(index)
            random.shuffle(indexes)
        else:
            mix_2_pos = None
            indexes = [index]

        c = 0
        mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list = [], [], [], []
        for idx in indexes:
            if self.mixup_cfg is not None:
                while True:
                    ratio = ratios[c]
                    c += 1
                    if 0 not in ratio:
                        break
            else:
                ratio = [1, 1]

            # Read image and annotations
            chip_name, ori_chip, ori_bbox, ori_label, ori_lm = self._prepare_ori_data(idx)

            # Image augmentation
            if self.phase == 'train':
                chip, bbox, label, lm = self.augmentation.train_augmentation(
                    ori_chip, ori_bbox, ori_label, ori_lm, ratio, self.train_bbox_iof_threshold, self.train_min_num_landmarks)
            else:
                chip, bbox, label, lm = self.augmentation.val_augmentations(
                    ori_chip, ori_bbox, ori_label, ori_lm)

            mix_chip_list.append(chip)
            mix_bbox_list.append(bbox)
            mix_lm_list.append(lm)
            mix_label_list.append(label)

        # Combine mixup images and labels
        mixed_chip, mixed_bbox, mixed_lm, mixed_label = combine_mixup_imgs_labels(
            mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list, mix_2_pos)
        # # Pad image to square
        # mixed_chip = self.augmentation._pad_to_square_with_size(
        #     mixed_chip, self.augmentation.training_size)
        # Norm annotation
        mixed_bbox, mixed_lm = self.augmentation._norm_annotation(
            mixed_chip, mixed_bbox, mixed_lm)
        # Norm image
        mixed_chip = self.augmentation._subtract_mean(mixed_chip)

        # Create focus mask
        mask, flattened_mask = self._prepare_mask(mixed_chip, mixed_bbox)

        # Create chip label base on all bboxes label
        chip_label = 1 if 1 in mixed_label else 0

        # Original data to visualize in validation phase
        ori_data = None
        if self.phase == 'test':
            ori_data = {
                'chip_name': chip_name,
                'chip': ori_chip,
                'bbox': ori_bbox,
                'label': ori_label,
                'lm': ori_lm,
                'mask': mask,
                'chip_label': chip_label
            }
        ##TODO: Normalize mixed_chip
        return mixed_chip.transpose(2, 0, 1), mixed_bbox, mixed_label, mixed_lm, flattened_mask, chip_label, ori_data

    def _prepare_mask(self, chip, bbox):
        """
        Prepare focus mask corresponding with new chip
        """
        mask_w = math.ceil(chip.shape[1] / self.focus_gen.stride)
        mask_h = math.ceil(chip.shape[0] / self.focus_gen.stride)
        mask = np.zeros((mask_h, mask_w)).astype(np.long)
        for bb in bbox:
            scaled_bb = bb * ([chip.shape[1], chip.shape[0]] * 2)
            mask = self.focus_gen.calculate_mask(
                scaled_bb[0], scaled_bb[1], scaled_bb[2], scaled_bb[3], mask)
        flattened_mask = mask.reshape(mask.shape[0] * mask.shape[1])
        return mask, flattened_mask
    
    def _prepare_ori_data(self, idx):
        """
        Load original data to process
        """
        chip_name = self.chip_list[idx]
        ori_chip = cv2.imread(os.path.join(self.root_path, chip_name))
        ori_bbox = self.bbox_list[idx]
        ori_label = self.label_list[idx]
        ori_lm = self.lm_list[idx]
        return chip_name, ori_chip, ori_bbox, ori_label, ori_lm


def focus_retina_collate(batch):
    '''
    Custom collate function for dealing with batches of images
    that have a different number of annotations (bboxes, labels, lm)
    '''
    imgs, targets, masks, chip_labels_list, ori_data_list = [], [], [], [], []
    for sample in batch:
        chip, bbox, label, lm, mask, chip_label, ori_data = sample

        imgs.append(torch.from_numpy(chip))

        target = np.concatenate((bbox, lm, label), axis=-1)  # 4+10+1=15
        targets.append(torch.from_numpy(target))
        masks.append(torch.from_numpy(mask))
        chip_labels_list.append(chip_label)
        ori_data_list.append(ori_data)
    ##TODO: Should we stack targets to tensor with same number of targets per sample
    return (torch.stack(imgs, 0), targets, torch.stack(masks, 0), chip_labels_list, ori_data_list)


