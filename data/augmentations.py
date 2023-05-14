'''
Data Augmentations
'''
import random

import cv2
import numpy as np
import albumentations.augmentations.functional as F
from albumentations.augmentations import CenterCrop, ColorJitter


class CenterCropExt(CenterCrop):
    '''
    Custom CenterCrop
    '''

    @property
    def targets(self):
        return {
            'image': self.apply,
            'bboxes': self.apply_to_bboxes,
            'landmarks': self.apply_to_landmarks,
        }

    def apply_to_landmarks(self, landmarks, rows, cols):
        crop_coords = F.get_center_crop_coords(rows, cols, self.height, self.width)
        x1, y1, _, _ = crop_coords
        landmarks[:, 0::2] = (landmarks[:, 0::2] * cols - x1) / self.width
        landmarks[:, 1::2] = (landmarks[:, 1::2] * rows - y1) / self.height
        return landmarks

    def filter_valid_anno(self, bbox, label, lm):
        max_len_bbox = bbox.shape[1] // 2
        valid_pos = np.sum((bbox < 0) | (bbox > 1), axis=1) < max_len_bbox
        valid_bbox, valid_lm = map(self._extract_and_assign, [bbox, lm], [valid_pos, valid_pos])
        valid_label = label[valid_pos]
        return valid_bbox, valid_lm, valid_label

    @staticmethod
    def _extract_and_assign(item, valid_pos):
        item = item[valid_pos]
        item[item < 0] = 0
        item[item > 1] = 1
        return item


class DetectionAugmentation():
    '''
    Data Augmentation for Object Detection
    '''

    def __init__(self,
                 training_size,
                 brighten_param,
                 contrast_param,
                 saturate_param,
                 hue_param,
                 resize_methods,
                 rgb_mean,
                 pre_scales,
                 use_albumentations,
                 test_interpolation=cv2.INTER_CUBIC):
        self.training_size = training_size
        self.contrast_param = contrast_param
        self.saturate_param = saturate_param
        if use_albumentations:
            self.brighten_param = (1 - brighten_param[0] / 255, 1 + brighten_param[1] / 255)
            self.hue_param = (hue_param[0] / 360, hue_param[1] / 360)
            self.distort_fn = ColorJitter(brightness=self.brighten_param,
                                                     contrast=self.contrast_param,
                                                     saturation=self.saturate_param,
                                                     hue=self.hue_param)
        else:
            self.brighten_param = brighten_param
            self.hue_param = hue_param
        self.resize_methods = resize_methods
        self.rgb_mean = rgb_mean  # BGR order
        self.pre_scales = pre_scales
        self.use_albumentations = use_albumentations
        self.test_interpolation = test_interpolation
        # Init CenterCrop instance without caring about crop size
        # because we will assign crop size depend on size of image we forward
        self.center_crop = CenterCropExt(None, None)

    def train_augmentations(self, ori_chip, ori_bbox, ori_label, ori_lm, ratio, bbox_iof_threshold, min_num_landmarks):
        '''
        Augmentation used in training process
        '''
        _ori_chip, _ori_bbox, _ori_label, _ori_lm = \
            ori_chip.copy(), ori_bbox.copy(), ori_label.copy(), ori_lm.copy()

        # Random crop
        crop_wh_ratio = ratio[0] / ratio[1]
        chip, bbox, label, lm = self._random_crop(
            _ori_chip, _ori_bbox, _ori_label, _ori_lm, crop_wh_ratio, bbox_iof_threshold, min_num_landmarks)

        # Distort
        if self.use_albumentations:
            chip = self.distort_fn(image=chip)['image']
        else:
            chip = self._distort(chip)

        # Pad
        if crop_wh_ratio == 1:
            chip = self._pad_to_square(chip)
        else:
            chip = self._pad_to_scale(chip, crop_wh_ratio)

        # Horizontal flip
        chip, bbox, lm = self._mirror(chip, bbox, lm)

        # Resize
        new_size = (round(self.training_size * ratio[0]),
                    round(self.training_size * ratio[1]))
        resize_method = random.choice(self.resize_methods)
        chip, bbox, lm = self._resize_img_and_annos(chip, bbox, lm, new_size, resize_method)

        return chip, bbox, label, lm

    def val_augmentations(self, ori_chip, ori_bbox, ori_label, ori_lm):
        '''
        Augmentation used in validation process
        '''
        _ori_chip, _ori_bbox, _ori_label, _ori_lm = \
            ori_chip.copy(), ori_bbox.copy(), ori_label.copy(), ori_lm.copy()

        # Norm annotation to use in our custom CenterCrop
        bbox, lm = self._norm_annotation(_ori_chip, _ori_bbox, _ori_lm)
        # Assign new width and height to center crop according to min size of chip
        self.center_crop.width = self.center_crop.height = min(*_ori_chip.shape[0:2])
        # Foward Center crop
        result = self.center_crop(image=_ori_chip, bboxes=bbox, landmarks=lm)
        chip, bbox, lm = result['image'], result['bboxes'], result['landmarks']
        bbox, lm, label = self.center_crop.filter_valid_anno(
            np.array(bbox), _ori_label, np.array(lm))
        # Denorm annotation to after used in our custom CenterCrop
        bbox, lm = self._denorm_annotation(chip, bbox, lm)

        # Prevent the case of empty-bbox during the validation process
        if len(bbox) == 0:
            chip = self._pad_to_square(_ori_chip)
            bbox, lm, label = _ori_bbox, ori_lm, ori_label

        # Resize
        new_size = (self.training_size, self.training_size)
        resize_method = self.test_interpolation
        chip, bbox, lm = self._resize_img_and_annos(chip, bbox, lm, new_size, resize_method)
        return chip, bbox, label, lm

    def _random_crop(self, img, bbox, label, lm, crop_wh_ratio, bbox_iof_threshold, min_num_landmarks):
        '''
        Crop a region from image randomly
        '''
        img_h, img_w, _ = img.shape
        short_side = min(img_w, img_h)

        if max(self.pre_scales) > 1:
            # Create x4 image
            x4_img = np.zeros((2*img_h, 2*img_w, 3), dtype=img.dtype)
            x4_img[:img_h, :img_w] += img
            x4_img[:img_h, img_w:] += img
            x4_img[img_h:, :img_w] += img
            x4_img[img_h:, img_w:] += img

            # Create x4 bboxes
            x4_bbox_1 = bbox.copy()
            x4_bbox_1[:, 0::2] += img_w

            x4_bbox_2 = bbox.copy()
            x4_bbox_2[:, 1::2] += img_h

            x4_bbox_3 = bbox.copy()
            x4_bbox_3[:, 0::2] += img_w
            x4_bbox_3[:, 1::2] += img_h

            x4_bbox = np.concatenate((bbox, x4_bbox_1, x4_bbox_2, x4_bbox_3), axis=0)

            # Create x4 landmarks
            valid_lm_mask = lm[:, 1] != -1
            x4_lm_1 = lm.copy()
            x4_lm_1[valid_lm_mask, 0::2] += img_w

            x4_lm_2 = lm.copy()
            x4_lm_2[valid_lm_mask, 1::2] += img_h

            x4_lm_3 = lm.copy()
            x4_lm_3[valid_lm_mask, 0::2] += img_w
            x4_lm_3[valid_lm_mask, 1::2] += img_h

            x4_lm = np.concatenate((lm, x4_lm_1, x4_lm_2, x4_lm_3), axis=0)

            # Create x4 labels
            x4_label = np.tile(label, (4, 1))


        for _ in range(250):
            # Prepare roi
            scale = random.choice(self.pre_scales)
            if scale <= 1:
                selected_img = img
                selected_bbox = bbox
                selected_lm = lm
                selected_label = label
                img_h, img_w, _ = img.shape
            else:
                selected_img = x4_img
                selected_bbox = x4_bbox
                selected_lm = x4_lm
                selected_label = x4_label
                img_h, img_w, _ = x4_img.shape
            if crop_wh_ratio > 1:
                crop_w = scale * short_side
                crop_h = crop_w / crop_wh_ratio
            else:
                crop_h = scale * short_side
                crop_w = crop_h * crop_wh_ratio
            crop_w, crop_h = int(crop_w), int(crop_h)

            delta_w = random.randrange(
                img_w - crop_w) if img_w != crop_w else 0
            delta_h = random.randrange(
                img_h - crop_h) if img_h != crop_h else 0
            roi = np.array(
                (delta_w, delta_h, delta_w + crop_w, delta_h + crop_h))

            # Find valid bboxes
            bbox_idx = self._get_valid_bbox_idx(selected_bbox, roi, selected_lm, selected_label, bbox_iof_threshold, min_num_landmarks)
            if not bbox_idx.any(): # If no bbox fully covered by roi or includes at least `n` landmarks
                continue
            bbox_idx = bbox_idx.squeeze(axis=1).copy()

            roi_bbox = selected_bbox[bbox_idx].copy()
            roi_label = selected_label[bbox_idx].copy()
            roi_lm = selected_lm[bbox_idx].copy()

            # Convert bbox coordinate in image to bbox coordinate in roi
            roi_bbox[:, :2] = np.maximum(roi_bbox[:, :2], roi[:2])
            roi_bbox[:, :2] -= roi[:2]
            roi_bbox[:, 2:] = np.minimum(roi_bbox[:, 2:], roi[2:])
            roi_bbox[:, 2:] -= roi[:2]

            # Convert lm coordinate in image to lm coordinate in roi
            roi_lm = roi_lm.reshape([-1, 5, 2])
            roi_lm[:, :, :2] = roi_lm[:, :, :2] - roi[:2]
            roi_lm[:, :, :2] = np.maximum(roi_lm[:, :, :2], np.array([0, 0]))
            roi_lm[:, :, :2] = np.minimum(roi_lm[:, :, :2], roi[2:] - roi[:2])
            roi_lm = roi_lm.reshape([-1, 10])

            # Crop image
            roi_img = selected_img[roi[1]:roi[3], roi[0]:roi[2]]
            return roi_img, roi_bbox, roi_label, roi_lm

        return img, bbox, label, lm

    def _distort(self, img):
        '''
        Distort image brightness, contrast, saturate and hue
        '''
        def _random_augment():
            contrast_first = random.randrange(2)
            brighten = random.randrange(2)
            contrast = random.randrange(2)
            saturate = random.randrange(2)
            distort_hue = random.randrange(2)

            return contrast_first, brighten, contrast, saturate, distort_hue

        def _distort(img, alpha=1, beta=0):
            tmp = img.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            img[:] = tmp
            return img

        def _hue_distort(img):
            tmp = img[:, :, 0].astype(int)
            tmp += random.randint(self.hue_param[0], self.hue_param[1])
            tmp %= self.hue_param[2]
            img[:, :, 0] = tmp
            return img

        img = img.copy()
        contrast_first, brighten, contrast, saturate, distort_hue = _random_augment()

        # prepare distort params
        brighten_b = random.uniform(
            self.brighten_param[0], self.brighten_param[1])
        contrast_a = random.uniform(
            self.contrast_param[0], self.contrast_param[1])
        saturate_a = random.uniform(
            self.saturate_param[0], self.saturate_param[1])

        # brightness distortion
        img = _distort(img, beta=brighten_b) if brighten else img

        # contrast distortion first or last
        if contrast_first:
            # contrast distortion
            img = _distort(img, alpha=contrast_a) if contrast else img
            # convert to HSV to distort saturation and hue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # saturation distortion
            img[:, :, 1] = _distort(
                img[:, :, 1], alpha=saturate_a) if saturate else img[:, :, 1]
            # hue distortion
            img = _hue_distort(img) if distort_hue else img
            # convert back to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            # convert to HSV to distort saturation and hue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # saturation distortion
            img[:, :, 1] = _distort(
                img[:, :, 1], alpha=saturate_a) if saturate else img[:, :, 1]
            # hue distortion
            img = _hue_distort(img) if distort_hue else img
            # convert back to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            # contrast distortion
            img = _distort(img, alpha=contrast_a) if contrast else img
        return img

    def _pad_to_square(self, img):
        '''
        If image is not square, pad it to square
        '''
        if img.shape[0] == img.shape[1]:
            return img
        img_h, img_w, _ = img.shape
        long_side = max(img_w, img_h)
        padded_img = np.empty((long_side, long_side, 3), dtype=img.dtype)
        padded_img[:, :] = self.rgb_mean
        padded_img[0:0 + img_h, 0:0 + img_w] = img
        return padded_img

    def _pad_to_square_with_size(self, img, size):
        '''
        If image is not square, pad it to square with input size
        '''
        img_h, img_w, _ = img.shape
        if img_h == size and img_w == size:
            return img
        padded_img = np.empty((size, size, 3), dtype=img.dtype)
        padded_img[:, :] = self.rgb_mean
        padded_img[0:0 + img_h, 0:0 + img_w] = img
        return padded_img

    def _pad_to_scale(self, img, crop_wh_ratio):
        '''
        Pad image to input crop width height ratio
        '''
        img_h, img_w, _ = img.shape
        img_wh_ratio = img_w / img_h

        if crop_wh_ratio == img_wh_ratio:
            return img

        new_img_h, new_img_w = img_h, img_w
        if (crop_wh_ratio > 1 and img_wh_ratio > 1) or \
           (crop_wh_ratio < 1 and img_wh_ratio < 1) or \
           img_wh_ratio == 1: # Same ratio-type
            if crop_wh_ratio > img_wh_ratio:
                new_img_w = round(img_h * crop_wh_ratio)
            else: # crop_wh_ratio < img_wh_ratio
                new_img_h = round(img_w / crop_wh_ratio)
        else: # Different ratio-type
            if img_h > img_w:
                new_img_w = round(img_h * crop_wh_ratio)
            else: # img_w > img_h:
                new_img_h = round(img_w / crop_wh_ratio)

        padded_img = np.empty((new_img_h, new_img_w, 3), dtype=img.dtype)
        padded_img[:, :] = self.rgb_mean
        padded_img[0:0 + img_h, 0:0 + img_w] = img
        return padded_img

    def _subtract_mean(self, img):
        '''
        Normalize image
        '''
        img = img.astype(np.float32)
        img -= self.rgb_mean
        return img
    
    @staticmethod
    def _resize_img_and_annos(chip, bbox, lm, new_size, interpolation):
        # Scale bbox
        bbox[:, 0::2] = bbox[:, 0::2] * new_size[0] / chip.shape[1]
        bbox[:, 1::2] = bbox[:, 1::2] * new_size[1] / chip.shape[0]
        # Scale lm
        lm[:, 0::2] = lm[:, 0::2] * new_size[0] / chip.shape[1]
        lm[:, 1::2] = lm[:, 1::2] * new_size[1] / chip.shape[0]
        # Scale image
        chip = cv2.resize(chip, new_size, interpolation=interpolation)
        return chip, bbox, lm

    @staticmethod
    def _mirror(img, bbox, lm):
        '''
        Horizontal flip
        '''
        if random.randrange(2):
            _, img_w, _ = img.shape
            # Flip image
            img = img[:, ::-1]
            # Flip bbox
            bbox = bbox.copy()
            bbox[:, 0::2] = img_w - bbox[:, 2::-2]
            # Flip lm
            lm = lm.copy()
            lm = lm.reshape([-1, 5, 2])
            lm[:, :, 0] = np.where(lm[:, :, 0] > 0, img_w - lm[:, :, 0], 0)
            tmp = lm[:, 1, :].copy()
            lm[:, 1, :] = lm[:, 0, :]
            lm[:, 0, :] = tmp
            tmp1 = lm[:, 4, :].copy()
            lm[:, 4, :] = lm[:, 3, :]
            lm[:, 3, :] = tmp1
            lm = lm.reshape([-1, 10])

        return img, bbox, lm

    @staticmethod
    def _norm_annotation(img, bbox, lm):
        '''
        Scale bbox and lm with new image scale
        '''
        img_h, img_w, _ = img.shape
        bbox[:, 0::2] /= img_w
        bbox[:, 1::2] /= img_h
        # Normalize lm
        lm[:, 0::2] /= img_w
        lm[:, 1::2] /= img_h
        return bbox, lm

    @staticmethod
    def _denorm_annotation(img, bbox, lm):
        '''
        Denormalize bbox from relative to absolute
        '''
        img_h, img_w, _ = img.shape
        bbox[:, 0::2] *= img_w
        bbox[:, 1::2] *= img_h
        # Normalize lm
        lm[:, 0::2] *= img_w
        lm[:, 1::2] *= img_h
        return bbox, lm

    @staticmethod
    def _matrix_iof(a, b):
        '''
        Calculate intersection over foreground
        '''
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / np.maximum(area_a[:, np.newaxis], 1)

    @staticmethod
    def _get_valid_landmark_idx(landmarks, roi):
        ''' Get index mask of all landmarks fully covered by roi
        '''
        x1y1_diff = (landmarks - roi[:2]) > 0
        x1y1_pass = np.logical_and(x1y1_diff[:, :, 0], x1y1_diff[:, :, 1])
        x2y2_diff = (roi[2:] - landmarks) > 0
        x2y2_pass = np.logical_and(x2y2_diff[:, :, 0], x2y2_diff[:, :, 1])

        return np.logical_and(x1y1_pass, x2y2_pass)

    @staticmethod
    def _get_valid_bbox_idx(bboxes, roi, landmarks, labels, bbox_iof_threshold, min_num_landmarks=3):
        ''' Get index mask of all valid bboxes
        '''
        # iof >= 1 means bbox is fully covered by roi
        iof = DetectionAugmentation._matrix_iof(bboxes, roi[np.newaxis])

        _labels = labels.reshape(-1)
        _landmarks = landmarks.reshape(-1, 5, 2)
        mr_idx = _labels == 1

        # Decide valid bboxes for MR class
        valid_landmark_idx = DetectionAugmentation._get_valid_landmark_idx(_landmarks, roi)
        valid_count_idx = np.sum(valid_landmark_idx, -1) >= min_num_landmarks
        mr_pass_idx = np.logical_and(mr_idx, valid_count_idx)
        mr_refuse_idx = np.logical_and(mr_idx, ~valid_count_idx)
        iof[mr_pass_idx] = 1.0 # Auto accept all MR bboxes have at least `n` landmarks
        iof[mr_refuse_idx] = 0.0 # Refuse all MR bboxes don't have enough landmarks

        return iof >= bbox_iof_threshold