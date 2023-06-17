'''
Generate focus mask of chip
'''
import cv2
import json
import math
from shapely.geometry import Polygon

from tqdm import tqdm
import numpy as np


class FocusGenerator():
    '''
    Generate focus mask of chip
    '''

    def __init__(self,
                 dont_care_low,
                 dont_care_high,
                 small_threshold,
                 stride):
        self.dont_care_low = dont_care_low
        self.dont_care_high = dont_care_high
        self.small_threshold = small_threshold
        ##TODO: Dynamic this
        kernel_size = 3
        self.dilation_kernel = np.ones((kernel_size, kernel_size)).astype(np.uint8)
        self.stride = stride

    def __call__(self, json_path, final_out):
        with open(json_path, 'r') as file:
            data = json.load(file)

        all_img_data = []
        for img_data in tqdm(data):
            chip_data_list = []
            for chip_data in img_data['image_chips']:
                label_size = math.ceil(chip_data['chip_size'] / self.stride)
                c_mask = np.zeros((label_size, label_size))
                bboxes_list = chip_data['chip_valid_bboxes'] + chip_data['chip_invalid_bboxes']
                for _, bbox_data in enumerate(bboxes_list):
                    c_mask = self.calculate_mask(bbox_data['bbox_x1'],
                                                bbox_data['bbox_y1'],
                                                bbox_data['bbox_x2'],
                                                bbox_data['bbox_y2'],
                                                c_mask)
                mask = c_mask.reshape(label_size * label_size)
                one_list = list(np.where(mask == 1)[0].astype(float))
                minus_one_list = list(np.where(mask == -1)[0].astype(float))

                new_chip_data = {
                    'chip_mask_len': label_size * label_size,
                    'one_index': one_list,
                    'minus_one_index': minus_one_list
                }
                chip_data.update(new_chip_data)
                chip_data_list.append(chip_data)
            img_data['image_chips'] = chip_data_list
            all_img_data.append(img_data)

        with open(final_out, 'w') as file:
            json.dump(all_img_data, file, indent=4, sort_keys=True)

    ##TODO: Delete this
    def calculate_mask(self, bbox_x1, bbox_y1, bbox_x2, bbox_y2, c_mask):
        area = np.sqrt((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1))

        x1 = int(bbox_x1 / self.stride)
        y1 = int(bbox_y1 / self.stride)
        x2 = int(math.ceil(bbox_x2 / self.stride))
        y2 = int(math.ceil(bbox_y2 / self.stride))

        ##TODO: Need change threshold for text
        if self.dont_care_low < area < self.small_threshold:
            flag = 1
        elif (self.small_threshold <= area < self.dont_care_high) or \
                (area <= self.dont_care_low):
            flag = -1
        else:
            flag = 0

        ##TODO: Substitue the masking code map
        for p1 in range(x1, min(x2 + 1, c_mask.shape[1])):
            for p2 in range(y1, min(y2 + 1, c_mask.shape[0])):
                if c_mask[p2][p1] != 1:
                    c_mask[p2][p1] = flag
        return c_mask

    def get_mask_flag_from_landmarks(self, lms):
        polygons = Polygon(lms)
        area_sqrt = np.sqrt(polygons.area)

        lms = lms.copy() / self.stride

        ##TODO: Need change threshold for text
        if self.dont_care_low < area_sqrt < self.small_threshold:
            return 1
        elif (self.small_threshold <= area_sqrt < self.dont_care_high) or \
                (area_sqrt <= self.dont_care_low):
            return -1
        else:
            return 0
    
    ##TODO: Change name
    def calculate_mask_by_landmarks_group(self, lms_group, c_mask):
        lms_dont_care_list = []
        lms_care_list = []
        for lms in lms_group:
            if self.get_mask_flag_from_landmarks(lms) == 1:
                lms_care_list.append(lms)
            elif self.get_mask_flag_from_landmarks(lms) == -1:
                lms_dont_care_list.append(lms)

        dont_care_mask = c_mask.copy().astype(np.uint8)
        care_mask = c_mask.copy().astype(np.uint8)

        cv2.fillPoly(dont_care_mask, [(lms.copy() / self.stride).astype(np.long) for lms in lms_dont_care_list], 1)
        # dont_care_mask = cv2.dilate(dont_care_mask, self.dilation_kernel)
        cv2.fillPoly(care_mask, [(lms.copy() / self.stride).astype(np.long) for lms in lms_care_list], 1)
        care_mask = cv2.dilate(care_mask, self.dilation_kernel)

        care_mask = care_mask.astype(int)
        care_mask[np.logical_and(dont_care_mask == 1, care_mask != 1)] = -1
        return care_mask


# if __name__ == "__main__":
#     args = focus_generator_parser()

#     focus_generator = FocusGenerator(args.dont_care_low,
#                                      args.dont_care_high,
#                                      args.small_threshold,
#                                      args.stride)
#     focus_generator(args.input_path, args.output_path)