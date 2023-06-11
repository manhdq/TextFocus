'''
Generate positive and negative chip
'''
import argparse
import os
import random
import sys
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


from chips import generate


def get_parser():
    '''
    Parse arguments of chip generator
    '''
    parser = argparse.ArgumentParser(
        description='Generating chip')
    parser.add_argument('--img-train-dir', type=str,
                        help='Path to training image file to generate chip')
    parser.add_argument('--img-val-dir', type=str,
                        help='Path to validating image file to generate chip')
    parser.add_argument('--img-test-dir', type=str,
                        help='Path to testing image file to generate chip')
    parser.add_argument('--ann-train-dir', type=str,
                        help='Path to training annotation file to generate chip')
    parser.add_argument('--ann-val-dir', type=str,
                        help='Path to validating annotation file to generate chip')
    parser.add_argument('--ann-test-dir', type=str,
                        help='Path to testing annotation file to generate chip')
    parser.add_argument('--chip-save-dir', type=str,
                        help='Path to output chips folder')
    parser.add_argument('--valid-range', type=int,
                        help='Valid range of size of bbox for each chip size')
    parser.add_argument('--c-stride', type=int,
                        help='Stride while sliding chip')
    parser.add_argument('--mapping-threshold', type=float,
                        help='Threshold to map our data to WIDERFACE data')
    parser.add_argument('--training-size', type=int,
                        help='Size of image to training detection')
    parser.add_argument('--n-threads', type=int,
                        help='Num of threads to run multi processing')
    parser.add_argument('--use-neg', type=int,
                        help='Generate negative chip or not')
    args = parser.parse_args()
    return args


class ChipGenerator():
    '''
    Generate positive and negative chip
    '''

    def __init__(self,
                valid_range,
                c_stride,
                mapping_threshold,
                training_size,
                n_threads=16,
                use_neg=False):
        self.valid_range = valid_range
        self.c_stride = c_stride
        self.mapping_threshold = mapping_threshold
        self.training_size = training_size
        self.n_threads = n_threads
        self.use_neg = use_neg

    def __call__(self, img_dirs, ann_dirs, chip_save_dir, save_data_type="yolo"):
        chip_img_dirs = []
        chip_ann_dirs = []
        # Make dir for saving
        if save_data_type == "yolo":
            for i, phase in enumerate(["train", "val", "test"]):
                if img_dirs[i] is None or ann_dirs[i] is None:
                    chip_img_dirs.append(None)
                    chip_ann_dirs.append(None)
                else:
                    chip_img_dir = os.path.join(chip_save_dir, "Images", f"chip_for_{phase}")
                    os.makedirs(chip_img_dir, exist_ok=True)
                    chip_img_dirs.append(chip_img_dir)
                    chip_ann_dir = os.path.join(chip_save_dir, "gt", f"chip_for_{phase}")
                    os.makedirs(chip_ann_dir, exist_ok=True)
                    chip_ann_dirs.append(chip_ann_dir)
        else:
            ##TODO:
            raise

        for phase, img_dir, ann_dir, chip_img_dir, chip_ann_dir in \
                            zip(["train", "val", "test"], img_dirs, ann_dirs, chip_img_dirs, chip_ann_dirs):
            if chip_img_dir is None:  # skip
                continue

            img_name_list = os.listdir(img_dir)
            img_name_list = [img_name.split('.')[0] for img_name in img_name_list]
            print(f"Generating chips from {len(img_name_list)} images for phase {phase}...")
            
            # for img_name in tqdm(img_name_list, total=len(img_name_list)):
            #     self._positive_gen(img_name, img_dir, ann_dir, chip_img_dir, chip_ann_dir)
            with Pool(self.n_threads) as pool:
                all_data = list(tqdm(pool.imap(partial(self._positive_gen, img_dir=img_dir, ann_dir=ann_dir,
                                        chip_img_dir=chip_img_dir, chip_ann_dir=chip_ann_dir), img_name_list),
                        total=len(img_name_list)))

    def _positive_gen(self, img_name, img_dir, ann_dir, chip_img_dir, chip_ann_dir):
        """
        Generate positive chips
        """
        img_path = os.path.join(img_dir, f"{img_name}.jpg")
        ann_path = os.path.join(ann_dir, f"{img_name}.txt")

        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size
        labels, boxes, kpts_list, texts = get_annotation_from_txt(ann_path, (img_w, img_h))
        
        # Calculate sqrt bbox areas
        ws = (boxes[:, 2] - boxes[:, 0]).astype(np.int32)
        hs = (boxes[:, 3] - boxes[:, 1]).astype(np.int32)
        sqrt_bb_areas = np.sqrt(ws * hs)

        # Calculate sqrt image area
        min_size = min(img_w, img_h)
        max_size = max(img_w, img_h)
        sqrt_img_area = np.sqrt(img_w * img_h)

        ori_img = cv2.imread(img_path)

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
        chips = self._generate_chip(boxes, (img_w, img_h), c_size, c_stride)

        # Get valid bboxes
        _valid_range = self.valid_range * sqrt_img_area / self.training_size
        valid_ids = np.where(sqrt_bb_areas >= _valid_range)[0]
        valid_bboxes = boxes[valid_ids, :]
        valid_kpts = [kpts_list[valid_id] for valid_id in valid_ids]
        chip_clses = labels[valid_ids]
        chip_texts = [texts[valid_id] for valid_id in valid_ids]

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
            chip_name = f"{img_name}_{chip_id}"
            chip_img_path = os.path.join(chip_img_dir, f"{chip_name}.jpg")
            cv2.imwrite(chip_img_path, padding_chip, [cv2.IMWRITE_JPEG_QUALITY, 100])

            chip_ann_path = os.path.join(chip_ann_dir, f"{chip_name}.txt")

            # Mapping valid bboxes and kpts with chip
            lines = []
            for j, (valid_bbox, valid_kpt, lbl, text) in enumerate(zip(valid_bboxes, valid_kpts, chip_clses, chip_texts)):
                if self.check_valid_bbox_and_kpts_in_chip(valid_bbox, valid_kpt, chip):
                    line = f"{lbl}"

                    chip_w = chip[2] - chip[0]
                    chip_h = chip[3] - chip[1]

                    chip_box_x1 = valid_bbox[0] - chip[0]
                    chip_box_y1 = valid_bbox[1] - chip[1]
                    chip_box_x2 = valid_bbox[2] - chip[0]
                    chip_box_y2 = valid_bbox[3] - chip[1]

                    chip_box_xn = (chip_box_x1 + chip_box_x2) / 2 / chip_w
                    chip_box_yn = (chip_box_y1 + chip_box_y2) / 2 / chip_h
                    chip_box_wn = (chip_box_x2 - chip_box_x1) / chip_w
                    chip_box_hn = (chip_box_y2 - chip_box_y1) / chip_h

                    line = line + f" {chip_box_xn:.4f} {chip_box_yn:.4f} {chip_box_wn:.4f} {chip_box_hn:.4f} "

                    chip_kpt = valid_kpt - np.array([chip[0], chip[1]])[None]
                    chip_kpt = chip_kpt.flatten()
                    line = line + " ".join(list(map(str, chip_kpt)))

                    line = line + f" | {text}"
                    lines.append(line)
            
            with open(chip_ann_path, "w") as f:
                f.write("\n".join(lines))

        return None, None

    @staticmethod
    def check_valid_bbox_and_kpts_in_chip(bbox, kpts, chip, area_threshold=0.01):
        """
        Check bbox and kpts in chip
        """
        # ##TODO: Maybe we need just kpts
        # w = int(chip[2] - chip[0])
        # h = int(chip[3] - chip[1])
        # sqrt_img_size = np.sqrt(w * h)
        # min_area_threshold = int(sqrt_img_size * area_threshold)

        # mask = np.zeros((h, w))
        # cv2.fillPoly(mask, [kpts.astype(int)], 1)
        # return np.sqrt(mask.sum()) >= min_area_threshold
        bbox = bbox.astype(int)
        return bbox[0] >= chip[0] and bbox[2] <= chip[2] and \
            bbox[1] >= chip[1] and bbox[3] <= chip[3]

    def _generate_chip(self, boxes, img_size, c_size, c_stride):
        '''
        Generate chips from bbox
        '''
        width, height = img_size
        clipped_bboxes = self.clip_bboxes(boxes, np.array([height - 1, width - 1]))
        chips = generate(np.ascontiguousarray(clipped_bboxes, dtype=np.float32),
                            width, height, c_size, c_stride)
        return chips

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


def get_annotation_from_txt(ann_path, img_size):
    w, h = img_size

    with open(ann_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    boxes = []
    labels = []
    kpts_list = []
    texts = []
    try:
        for line in lines:
            ann_text_infos = line.split("|")
            ann_infos = ann_text_infos[0]
            text = "|".join(ann_text_infos[1:])
            text = text.strip()
            texts.append(text)
            ann_infos = ann_infos.strip().split()
            
            lbl = int(ann_infos[0])
            labels.append(lbl)

            xn, yn, wn, hn = tuple(map(float, ann_infos[1:5]))
            x1 = (xn - wn / 2) * w
            y1 = (yn - hn / 2) * h
            x2 = (xn + wn / 2) * w
            y2 = (yn + hn / 2) * h
            boxes.append([x1, y1, x2, y2])

            kpts = np.array(list(map(float, ann_infos[5:]))).reshape((-1, 2))
            kpts_list.append(kpts)
        
        boxes = np.array(boxes)
        labels = np.array(labels)

        return labels, boxes, kpts_list, texts
    
    except:
        print(ann_path)
        raise


if __name__ == "__main__":
    args = get_parser()

    test = "739 | 2008 GENERAL MILLS IBERICA, SAU Reservados todos los derechos. Informacion legal | Aviso legal | Politics de proteccion de datos | Mapa Weo"
    
    chip_generator = ChipGenerator(args.valid_range,
                                args.c_stride,
                                args.mapping_threshold,
                                args.training_size,
                                args.n_threads,
                                args.use_neg)

    img_dirs = [args.img_train_dir, args.img_val_dir, args.img_test_dir]
    ann_dirs = [args.ann_train_dir, args.ann_val_dir, args.ann_test_dir]

    chip_generator(img_dirs, ann_dirs, args.chip_save_dir)