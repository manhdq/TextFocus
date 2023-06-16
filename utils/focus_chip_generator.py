'''
FocusChip Generator algorithm in Inference and Demo phase
'''
import math

import cv2
import numpy as np

class FocusChip():
    '''
    FocusChip Generator algorithm in Inference phase
    '''
    def __init__(self, threshold, kernel_size, min_chip_size, stride):
        self.threshold = threshold
        self.dilation_kernel = np.ones((kernel_size, kernel_size)).astype(np.uint8)
        self.min_mask_crop_size = int(min_chip_size / stride)
        self.stride = stride

    def __call__(self, pred_mask, img_w, img_h):
        # Transform pred_mask into binary
        binary_mask = np.where(pred_mask > self.threshold, 1, 0).astype(np.uint8)
        # Dilate with dilation kernel
        dilated_mask = cv2.dilate(binary_mask, self.dilation_kernel)
        mask = (dilated_mask * 255).astype(np.uint8)
        mask_h, mask_w = pred_mask.shape

        # Trun mask into rectangle mask
        crops = []
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            # Get smallest rectangle that cover contour
            rec_x, rec_y, rec_w, rec_h = cv2.boundingRect(cnt)
            rec_cx = rec_x + rec_w / 2
            rec_cy = rec_y + rec_h / 2
            # Get the crop size in mask
            crop_w = max(self.min_mask_crop_size, rec_w)
            crop_h = max(self.min_mask_crop_size, rec_h)
            # Check if 0 < crop width < mask_width
            if rec_cx + crop_w / 2 >= mask_w:
                crop_x = mask_w - crop_w if mask_w - crop_w >= 0 else 0
            elif rec_cx - crop_w / 2 < 0:
                crop_x = 0
            else:
                crop_x = np.ceil(rec_cx - crop_w / 2).astype(np.int)
            # Check if 0 < crop height < mask height
            if rec_cy + crop_h / 2 >= mask_h:
                crop_y = mask_h - crop_h if mask_h - crop_h >= 0 else 0
            elif rec_cy - crop_h / 2 < 0:
                crop_y = 0
            else:
                crop_y = np.ceil(rec_cy - crop_h / 2).astype(np.int)
            # Transform crop in to binary
            mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = 255
            crops.append([crop_x, crop_y, crop_x + crop_w, crop_y + crop_h])

        infer_scale = img_w / mask_w
        chips = []
        for crop in crops:
            chip = [c * infer_scale for c in crop]
            # Check chip_x2 in image
            if chip[2] > img_w:
                chip[2] = img_w
                # Check chip_x1 in image and chip_w > self.min_mask_crop_size
                chip[0] = max(min(chip[0], chip[2] - self.min_mask_crop_size), 0)
            # Check chip_y2
            if chip[3] > img_h:
                chip[3] = img_h
                chip[1] = max(min(chip[1], chip[3] - self.min_mask_crop_size), 0)
            # Convert chip coordinate back to image coordinate
            chips.append(np.array([int(c * 1) for c in chip]))
        return chips