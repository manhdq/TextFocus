'''
Utils for Mixup
'''
import random

import cv2
import numpy as np


def _stack_anno(mix_bbox_list, mix_lm_list, mix_label_list):
    '''
    Stack all annotations of all mixup images into one
    '''
    mixed_bbox = np.vstack(mix_bbox_list)
    mixed_lm = np.vstack(mix_lm_list)
    mixed_label = np.vstack(mix_label_list)
    return mixed_bbox, mixed_lm, mixed_label


def _combine_two_imgs_left_right(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list):
    '''
    Mixup two images (1 top, 1 bottom)
    '''
    # Concat image
    mixed_chip = cv2.hconcat([mix_chip_list[0], mix_chip_list[1]])
    # Convert bbox coordinate
    mix_bbox_list[1][:, 0::2] = mix_bbox_list[1][:, 0::2] + mix_chip_list[0].shape[1]
    # Convert lm coordinate
    mix_lm_list[1][:, 0::2] = np.where(
        mix_lm_list[1][:, 0::2] > 0, mix_lm_list[1][:, 0::2] + mix_chip_list[0].shape[1], 0)

    mixed_bbox, mixed_lm, mixed_label = \
        _stack_anno(mix_bbox_list, mix_lm_list, mix_label_list)
    return mixed_chip, mixed_bbox, mixed_lm, mixed_label


def _combine_two_imgs_up_down(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list):
    '''
    Mixup two images (1 left, 1 right)
    '''
    # Concat image
    mixed_chip = cv2.vconcat([mix_chip_list[0], mix_chip_list[1]])
    # Convert bbox coordinate
    mix_bbox_list[1][:, 1::2] = mix_bbox_list[1][:, 1::2] + mix_chip_list[0].shape[0]
    # Convert lm coordinate
    mix_lm_list[1][:, 1::2] = np.where(
        mix_lm_list[1][:, 1::2] > 0, mix_lm_list[1][:, 1::2] + mix_chip_list[0].shape[0], 0)

    mixed_bbox, mixed_lm, mixed_label = \
        _stack_anno(mix_bbox_list, mix_lm_list, mix_label_list)
    return mixed_chip, mixed_bbox, mixed_lm, mixed_label


def _combine_four_imgs(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list):
    '''
    Mixup four images (2x2)
    '''
    # Concat image
    mixed_chip_1 = cv2.hconcat([mix_chip_list[0], mix_chip_list[1]])
    mixed_chip_2 = cv2.hconcat([mix_chip_list[2], mix_chip_list[3]])
    mixed_chip = cv2.vconcat([mixed_chip_1, mixed_chip_2])
    # Convert bbox coordinate
    mix_bbox_list[1][:, 0::2] = mix_bbox_list[1][:, 0::2] + mix_chip_list[0].shape[1]
    mix_bbox_list[2][:, 1::2] = mix_bbox_list[2][:, 1::2] + mix_chip_list[0].shape[0]
    mix_bbox_list[3][:, 1::2] = mix_bbox_list[3][:, 1::2] + mix_chip_list[0].shape[0]
    mix_bbox_list[3][:, 0::2] = mix_bbox_list[3][:, 0::2] + mix_chip_list[0].shape[1]
    # Convert lm coordinate
    mix_lm_list[1][:, 0::2] = np.where(
        mix_lm_list[1][:, 0::2] > 0, mix_lm_list[1][:, 0::2] + mix_chip_list[0].shape[1], 0)
    mix_lm_list[2][:, 1::2] = np.where(
        mix_lm_list[2][:, 1::2] > 0, mix_lm_list[2][:, 1::2] + mix_chip_list[0].shape[0], 0)
    mix_lm_list[3][:, 1::2] = np.where(
        mix_lm_list[3][:, 1::2] > 0, mix_lm_list[3][:, 1::2] + mix_chip_list[0].shape[0], 0)
    mix_lm_list[3][:, 0::2] = np.where(
        mix_lm_list[3][:, 0::2] > 0, mix_lm_list[3][:, 0::2] + mix_chip_list[0].shape[1], 0)

    mixed_bbox, mixed_lm, mixed_label = \
        _stack_anno(mix_bbox_list, mix_lm_list, mix_label_list)
    return mixed_chip, mixed_bbox, mixed_lm, mixed_label


def _combine_one_img(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list):
    '''
    Mixup one image means return that image and its annotation
    '''
    mixed_chip = mix_chip_list[0]
    mixed_bbox, mixed_lm, mixed_label = \
        _stack_anno(mix_bbox_list, mix_lm_list, mix_label_list)
    return mixed_chip, mixed_bbox, mixed_lm, mixed_label


def combine_mixup_imgs_labels(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list, mix_2_pos):
    '''
    Mixup images and labels into one
    '''
    if len(mix_chip_list) == 2:
        if mix_2_pos == 'up_down':
            mixed_chip, mixed_bbox, mixed_lm, mixed_label = \
                _combine_two_imgs_up_down(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list)
        elif mix_2_pos == 'left_right':
            mixed_chip, mixed_bbox, mixed_lm, mixed_label = \
                _combine_two_imgs_left_right(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list)
    elif len(mix_chip_list) == 4:
        mixed_chip, mixed_bbox, mixed_lm, mixed_label = \
            _combine_four_imgs(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list)
    elif len(mix_chip_list) == 1:
        mixed_chip, mixed_bbox, mixed_lm, mixed_label = \
            _combine_one_img(mix_chip_list, mix_bbox_list, mix_lm_list, mix_label_list)
    return mixed_chip, mixed_bbox, mixed_lm, mixed_label


def prepare_mixup_params(cfg):
    '''
    Prepare Mixup parameters
    '''
    mix_type = random.choice(cfg['types'])
    mix_2_pos = None
    if mix_type == 2:
        mix_2_pos = random.choice(cfg['mix_2_pos_types'])
        if mix_2_pos == 'left_right':
            w_ratio, h_ratio = random.uniform(cfg['range'][0], cfg['range'][1]), 1
        else:
            w_ratio, h_ratio = 1, random.uniform(cfg['range'][0], cfg['range'][1])
    elif mix_type == 4:
        w_ratio = random.uniform(cfg['range'][0], cfg['range'][1])
        h_ratio = random.uniform(cfg['range'][0], cfg['range'][1])
    else:
        w_ratio = h_ratio = 1
    ratios = [(w_ratio, h_ratio), (1 - w_ratio, h_ratio),
              (w_ratio, 1 - h_ratio), (1 - w_ratio, 1 - h_ratio)]
    return mix_type, mix_2_pos, ratios
