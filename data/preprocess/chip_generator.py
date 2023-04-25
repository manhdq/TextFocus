'''
Generate positive and negative chip
'''
import json
import os
import random
import sys
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.append('.')
from utils.parser import chip_generator_parser

from chips import generate


class ChipGenerator():
    '''
    Generate positive and negative chip
    '''

    def __init__(self,
                 root_path,
                 ori_img_test_path,
                 ori_json_test_path,
                 valid_range,
                 c_stride,
                 mapping_threshold,
                 training_size,
                 n_threads=16,
                 use_neg=False):
        self.root_path = root_path
        self.ori_img_test_path = ori_img_test_path
        self.ori_json_test_path = ori_json_test_path
        self.valid_range = valid_range
        self.c_stride = c_stride
        self.mapping_threshold = mapping_threshold
        self.training_size = training_size    # RetinaFace input training size
        self.n_threads = n_threads
        self.use_neg = use_neg

    def __call__(self, coco_paths, out_json_paths, save_path):
        new_coco_paths = []
        new_json_paths = []
        phases = []

        for i, phase in enumerate(['train', 'val', 'test']):
            if coco_paths[i] is not None:
                phases.append(phase)
                new_coco_paths.append(coco_paths[i])
                new_json_paths.append(out_json_paths[i])
        
        coco_paths = new_coco_paths
        out_json_paths = new_json_paths

        chip_save_paths = [os.path.join(save_path, phase) for phase in phases]

        for i, (coco_path, out_json_path, chip_save_path) \
            in enumerate(zip(coco_paths, out_json_paths, chip_save_paths)):
            self.coco = COCO(coco_path)
            i_ids = self.coco.getImgIds()

            self.chip_save_path = chip_save_path
            if not os.path.exists(self.chip_save_path):
                os.makedirs(self.chip_save_path)

            print(f'Processing {coco_path} ...')
            print(f'Generating positive chips from {len(i_ids)} images')
            all_data = []
            ##TODO: Dont know such item cant be used, set rule later
            remove_list = [220, 424, 532, 1052]
            for remove_id in remove_list:
                if remove_id in i_ids:
                    i_ids.remove(remove_id)
            # for i_id in tqdm(i_ids, total=len(i_ids)):
            #     if i_id in [220, 424, 532, 1052]:
            #         continue
            #     all_data.append(self._positive_gen(i_id, phase=phases[i]))
            with Pool(self.n_threads) as pool:
                all_data = list(tqdm(pool.imap(partial(self._positive_gen, phase=phases[i]), i_ids),
                                total=len(i_ids)))
            all_img_data, all_ori_img_data = [], []
            for data in all_data:
                if data[0] is not None:
                    all_img_data.append(data[0])
                if data[1] is not None:
                    all_ori_img_data.append(data[1])

            with open(out_json_path, 'w') as file:
                json.dump(all_img_data, file, indent=4, sort_keys=True)
            
            if phases[i] == 'test':
                with open(self.ori_json_test_path, 'w') as file:
                    json.dump(all_ori_img_data, file, indent=4, sort_keys=True)

            if self.use_neg:
                print(f'Generating negative chips from {len(i_ids)} images')
                self._negative_gen()

    def _positive_gen(self, i_id, phase):
        '''
        Generate positive chips
        '''
        a_ids = self.coco.getAnnIds(imgIds=i_id)
        a_data = self.coco.loadAnns(a_ids)
        if len(a_data) > 0:
            i_data = self.coco.loadImgs(i_id)[0]
            bboxes = np.zeros((len(a_data), 4))
            clses = np.zeros((len(a_data), 1))
            for i, data in enumerate(a_data):
                bboxes[i, :] = data['clean_bbox']
                clses[i, :] = data['category_id']

            # Calculate sqrt bbox areas
            ws = (bboxes[:, 2] - bboxes[:, 0]).astype(np.int32)
            hs = (bboxes[:, 3] - bboxes[:, 1]).astype(np.int32)
            sqrt_bb_areas = np.sqrt(ws * hs)

            # Calculate sqrt image area
            img_w, img_h = i_data['width'], i_data['height']
            min_size = min(img_w, img_h)
            max_size = max(img_w, img_h)
            sqrt_img_area = np.sqrt(img_w * img_h)
            img_name = i_data['file_name'].split('/')[-1]

            ori_img_path = os.path.join(self.root_path, img_name)
            ori_img = cv2.imread(ori_img_path)
            if phase == 'test':
                os.system(f'sudo cp {ori_img_path} {self.ori_img_test_path}')
                ori_img_bb_list = []
                bb_id = 0
                for bb, cls in zip(bboxes, clses):
                    ori_img_bb_list.append({
                        'bbox_id': bb_id,
                        'bbox_x1': int(bb[0]),
                        'bbox_y1': int(bb[1]),
                        'bbox_x2': int(bb[2]),
                        'bbox_y2': int(bb[3]),
                        'class': int(cls[0])
                    })
                    bb_id += 1
                ori_img_data = {
                    'image_id': i_id,
                    'image_name': i_data['file_name'],
                    'image_width': i_data['width'],
                    'image_height': i_data['height'],
                    'image_chips': {
                        'chip_id': 0,
                        'chip_name': i_data['file_name'],
                        'chip_size': (i_data['width'], i_data['height']),
                        'chip_stride': 0,
                        'valid_range': 0,
                        'chip_x1': 0,
                        'chip_y1': 0,
                        'chip_x2': i_data['width'] - 1,
                        'chip_y2': i_data['height'] - 1,
                        'chip_valid_bboxes': ori_img_bb_list,
                        'chip_invalid_bboxes': []
                    }
                }

            # Calculate chip size
            if (max(sqrt_bb_areas) / sqrt_img_area) <= self.mapping_threshold:
                c_size = int(max(sqrt_bb_areas) / random.uniform(0.2, 0.4))
                # self.training_size <= c_size <= min_size
                c_size = max(c_size, self.training_size)
                c_size = min(c_size, min_size)
            else:
                c_size = min_size

            # Get chips from all bboxes
            c_stride = min((max_size - c_size) // 4, c_size // 2)
            chips = self._generate_chip(bboxes, img_w, img_h, c_size, c_stride)

            # Get valid bboxes
            _valid_range = self.valid_range * sqrt_img_area / self.training_size
            valid_ids = np.where(sqrt_bb_areas >= _valid_range)[0]
            valid_bboxes = bboxes[valid_ids, :]
            chip_clses = clses[valid_ids, :]

            # Get invalid bboxes
            # invalid_ids = list(set(range(len(a_ids))) - set(valid_ids))
            invalid_ids = np.where(sqrt_bb_areas < _valid_range)[0]
            invalid_bboxes = bboxes[invalid_ids, :]

            chip_id = 0
            chip_data_list = []
            for chip in chips:
                # Save chip to file
                output_chip = ori_img[int(chip[1]):int(chip[3]),
                                      int(chip[0]):int(chip[2]), :]
                padding_chip = np.zeros((c_size, c_size, output_chip.shape[2]))

                d1m = min(output_chip.shape[0], c_size)
                d2m = min(output_chip.shape[1], c_size)

                for j in range(3):
                    padding_chip[:d1m, :d2m, j] = output_chip[:d1m, :d2m, j]

                chip_id += 1
                chip_name = i_data['file_name'].split('/')[-1].split('.jpg')[0]
                chip_name = f'{chip_name}_{chip_id}.jpg'
                chip_path = os.path.join(self.chip_save_path, chip_name)
                cv2.imwrite(chip_path, padding_chip, [cv2.IMWRITE_JPEG_QUALITY, 100])

                # Mapping valid bboxes with chip
                valid_bbox_data_list = []
                valid_bbox_id = 0
                for j, valid_bbox in enumerate(valid_bboxes):
                    if self.check_valid_bbox_in_chip(valid_bbox, chip):
                        valid_bbox_id += 1
                        valid_bbox_data_list.append({
                            'bbox_id': valid_bbox_id,
                            'bbox_x1': int(valid_bbox[0]) - int(chip[0]),
                            'bbox_y1': int(valid_bbox[1]) - int(chip[1]),
                            'bbox_x2': int(valid_bbox[2]) - int(chip[0]),
                            'bbox_y2': int(valid_bbox[3]) - int(chip[1]),
                            'class': int(chip_clses[j])
                        })

                # Mapping invalid bboxes with chip
                invalid_bbox_data_list = []
                invalid_bbox_id = 0
                for j, invalid_bbox in enumerate(invalid_bboxes):
                    if self.check_invalid_bbox_in_chip(invalid_bbox, chip):
                        invalid_bbox_id += 1
                        invalid_bbox_data_list.append({
                            'bbox_id': invalid_bbox_id,
                            'bbox_x1': max(0, int(invalid_bbox[0]) - int(chip[0])),
                            'bbox_y1': max(0, int(invalid_bbox[1]) - int(chip[1])),
                            'bbox_x2': min(int(invalid_bbox[2]) - int(chip[0]),
                                           int(chip[2] - chip[0])),
                            'bbox_y2': min(int(invalid_bbox[3]) - int(chip[1]),
                                           int(chip[3] - chip[1])),
                        })

                chip_data_list.append({
                    'chip_id': chip_id,
                    'chip_name': chip_name,
                    'chip_size': c_size,
                    'chip_stride': c_stride,
                    'valid_range': _valid_range,
                    'chip_x1': int(chip[0]),
                    'chip_y1': int(chip[1]),
                    'chip_x2': int(chip[2]),
                    'chip_y2': int(chip[3]),
                    'chip_valid_bboxes': valid_bbox_data_list,
                    'chip_invalid_bboxes': invalid_bbox_data_list
                })

            img_data = {
                'image_id': i_id,
                'image_name': i_data['file_name'],
                'image_width': i_data['width'],
                'image_height': i_data['height'],
                'image_chips': chip_data_list
            }
            if phase == 'test':
                return img_data, ori_img_data
            return img_data, None
        return None, None

    def _negative_gen(self):
        '''
        Generate negative chips
        '''
        return

    def _generate_chip(self, boxes, width, height, c_size, c_stride):
        '''
        Generate chips from bbox
        '''
        clipped_bboxes = self.clip_bboxes(boxes, np.array([height - 1, width - 1]))
        chips = generate(np.ascontiguousarray(clipped_bboxes, dtype=np.float32),
                            width, height, c_size, c_stride)
        return chips

    @staticmethod
    def check_valid_bbox_in_chip(bbox, chip):
        '''
        Check bbox in chip
        '''
        bbox = bbox.astype(int)
        return bbox[0] >= chip[0] and bbox[2] <= chip[2] and \
            bbox[1] >= chip[1] and bbox[3] <= chip[3]

    @staticmethod
    def check_invalid_bbox_in_chip(i_bbox, chip):
        '''
        Check invalid bbox overlap a chip
        '''
        i_bbox = i_bbox.astype(int)
        xx_1, yy_1 = max(i_bbox[0], chip[0]), max(i_bbox[1], chip[1])
        xx_2, yy_2 = min(i_bbox[2], chip[2]), min(i_bbox[3], chip[3])

        if xx_1 < xx_2 and yy_1 < yy_2:
            return True
        return False

    @staticmethod
    def clip_bboxes(boxes, im_shape):
        '''
        Clip bboxes to image boundaries
        '''
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes


if __name__ == "__main__":
    args = chip_generator_parser()

    if not os.path.exists(args.ori_img_test_path):
        os.makedirs(args.ori_img_test_path)
    chip_generator = ChipGenerator(args.root_path,
                                   args.ori_img_test_path,
                                   args.ori_json_test_path,
                                   args.valid_range,
                                   args.c_stride,
                                   args.mapping_threshold,
                                   args.training_size,
                                   args.n_threads,
                                   args.use_neg)
    input_paths = [args.input_train_path, args.input_val_path, args.input_test_path]
    out_paths = [args.out_train_path, args.out_val_path, args.out_test_path]
    chip_generator(input_paths, out_paths, args.chip_save_path)
