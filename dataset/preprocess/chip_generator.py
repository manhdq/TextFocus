'''
Generate positive and negative chip
'''
import argparse
import os
import random
import sys
from functools import partial
from multiprocessing import Pool
from shapely.geometry import Polygon, Point, MultiPolygon, \
                        LineString, MultiLineString, GeometryCollection, \
                        MultiPoint

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


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


def points2box(points):
    x1 = points[:, 0].min()
    y1 = points[:, 1].min()
    x2 = points[:, 0].max()
    y2 = points[:, 1].max()
    return np.array((x1, y1, x2, y2), dtype=points.dtype)


class TextInstance(object):
    def __init__(self, points, orient, text, is_valid):
        self.orient = orient
        self.text = text
        self.is_valid = is_valid
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        """
        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)
        """
        self.points = np.array(points)
        self.box = points2box(self.points)

    def __repr__(self,):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


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
            ##TODO: Dont know such item cant be used, set rule later
            remove_list = ["0070", "0030", 
                        "1065", "1171", "1006", "1038", "1323"]
            for remove_id in remove_list:
                if remove_id in img_name_list:
                    img_name_list.remove(remove_id)

            for img_name in tqdm(img_name_list[0:], total=len(img_name_list)):
                print(img_name)
                self._positive_gen(img_name, img_dir, ann_dir, chip_img_dir, chip_ann_dir)
            # with Pool(self.n_threads) as pool:
            #     all_data = list(tqdm(pool.imap(partial(self._positive_gen, img_dir=img_dir, ann_dir=ann_dir,
            #                             chip_img_dir=chip_img_dir, chip_ann_dir=chip_ann_dir), img_name_list),
            #             total=len(img_name_list)))

    def _positive_gen(self, img_name, img_dir, ann_dir, chip_img_dir, chip_ann_dir):
        """
        Generate positive chips
        """
        img_path = os.path.join(img_dir, f"{img_name}.jpg")
        ann_path = os.path.join(ann_dir, f"{img_name}.txt")

        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size
        labels, obj_is_valids, boxes, kpts_list, texts = get_annotation_from_txt(ann_path, (img_w, img_h))
        
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
        valid_obj_is_valids = obj_is_valids[valid_ids]
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
            for j, (valid_bbox, valid_kpt, lbl, obj_is_valid, text) in enumerate(zip(valid_bboxes, valid_kpts, chip_clses, valid_obj_is_valids, chip_texts)):
                bboxes, kpts, obj_chip_is_valids = self.check_valid_bbox_and_kpts_in_chip(valid_bbox, valid_kpt, chip)

                if kpts is None or len(kpts)==0:  # Out of boundary
                    continue
                
                for kpt, bbox, obj_chip_is_valid in zip(kpts, bboxes, obj_chip_is_valids):
                    if kpt.shape[0] == 0:  # Out of boundary
                        continue

                    final_is_valid = obj_is_valid * obj_chip_is_valid

                    line = f"{lbl} {final_is_valid}"

                    chip_w = chip[2] - chip[0]
                    chip_h = chip[3] - chip[1]

                    chip_box_x1 = bbox[0] - chip[0]
                    chip_box_y1 = bbox[1] - chip[1]
                    chip_box_x2 = bbox[2] - chip[0]
                    chip_box_y2 = bbox[3] - chip[1]

                    chip_box_xn = (chip_box_x1 + chip_box_x2) / 2 / chip_w
                    chip_box_yn = (chip_box_y1 + chip_box_y2) / 2 / chip_h
                    chip_box_wn = (chip_box_x2 - chip_box_x1) / chip_w
                    chip_box_hn = (chip_box_y2 - chip_box_y1) / chip_h

                    line = line + f" {chip_box_xn:.4f} {chip_box_yn:.4f} {chip_box_wn:.4f} {chip_box_hn:.4f} "

                    chip_kpt = kpt - np.array([chip[0], chip[1]])[None]
                    chip_kpt = chip_kpt.flatten()
                    line = line + " ".join(list(map(str, chip_kpt)))

                    line = line + f" | {text}"
                    lines.append(line)
            
            ##TODO: 
            # polygons = self.parse_carve_lines(lines)
            # show_img = self.visualize_gt(np.array(padding_chip), polygons)
            # cv2.imwrite("test_chip.jpg", show_img[..., ::-1])
            # exit()
            
            with open(chip_ann_path, "w") as f:
                f.write("\n".join(lines))

        return None, None

    def parse_carve_lines(self, lines):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        
        polygons = []
        for line in lines:
            ann_infos = line.split(" | ")
            text = ann_infos[1:]
            text = " | ".join(text).strip()
            ann_infos = ann_infos[0].strip().split()
            is_valid = int(ann_infos[1])
            gt = list(map(float, ann_infos[6:]))
            assert len(gt) % 2 == 0
            pts = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)
            polygons.append(TextInstance(pts, "c", text, is_valid))

        return polygons

    ##TODO:
    def visualize_gt(self, image, contours, focus_mask=None):
    
        image_show = image.copy()
        h, w = image_show.shape[:2]
        image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min()) * 255
        image_show = np.ascontiguousarray(image_show[:, :, ::-1]).astype(np.uint8)

        # Add focus mask
        if focus_mask is not None:
            focus_mask = cv2.resize(focus_mask.copy(), (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = image_show.copy()
            mask_color_pos = (0, 255, 0)  # cyan
            mask_color_neg = (0, 0, 255)  # red
            overlay[:, :, 0][focus_mask == 1] = mask_color_pos[0] # Blue
            overlay[:, :, 1][focus_mask == 1] = mask_color_pos[1] # Green
            overlay[:, :, 2][focus_mask == 1] = mask_color_pos[2] # Red
            overlay[:, :, 0][focus_mask == -1] = mask_color_neg[0] # Blue
            overlay[:, :, 1][focus_mask == -1] = mask_color_neg[1] # Green
            overlay[:, :, 2][focus_mask == -1] = mask_color_neg[2] # Red
            image_show = cv2.addWeighted(overlay, 0.5, image_show, 0.5, 0)

        for contour in contours:
            box = contour.box.astype(int)
            image_show = cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]),
                                    (255, 0, 0), 1)

            boundary_color = (0, 255, 0) if contour.is_valid else (0, 0, 255)
            image_show = cv2.polylines(image_show,
                                    [contour.points.astype(int)], True, boundary_color, 2)

        # show_gt = cv2.resize(image_show, (320, 320))

        return image_show

    ##TODO: Make `area_threshold` dynamic. Delete `bbox`
    def check_valid_bbox_and_kpts_in_chip(self, bbox, kpts, chip, area_threshold=0.6):
        """
        Check bbox and kpts in chip
        """
        chip_kpts = self.bbox2kpts(chip)
        chip_polygon = Polygon(chip_kpts)
        obj_polygon = Polygon(kpts)
        ori_area = obj_polygon.area
        inter_polygon = obj_polygon.intersection(chip_polygon)
        
        inter_kpts = []
        if isinstance(inter_polygon, Polygon):
            inter_kpts = [np.array(inter_polygon.exterior.coords[:-1], dtype=np.int32)]  ##TODO: We need int??
        elif isinstance(inter_polygon, (Point, MultiPoint, LineString, MultiLineString)):
            return None, None, False
        elif isinstance(inter_polygon, MultiPolygon):
            inter_kpts = [np.array(inter_part_polygon.exterior.coords[:-1], dtype=np.int32) for \
                            inter_part_polygon in inter_polygon]
        elif isinstance(inter_polygon, GeometryCollection):
            for inter_part_polygon in inter_polygon:
                if isinstance(inter_part_polygon, Polygon):
                    inter_kpts.append(np.array(inter_part_polygon.exterior.coords[:-1], dtype=np.int32))
        else:
            ##TODO: Raise
            print(inter_polygon)
            inter_kpts = [np.array(inter_polygon.exterior.coords[:-1], dtype=np.int32)]

        ## Get box and check valid
        bboxes = []
        is_valids = []
        inter_kpts_out = []
        for inter_kpt in inter_kpts:
            if inter_kpt.shape[0] == 0:
                continue

            inter_area = Polygon(inter_kpt).area
            area_ratio = inter_area / ori_area
            # Get box
            bboxes.append(self.kpts2bbox(inter_kpt))

            if area_ratio >= area_threshold:
                is_valids.append(True)
            else:
                is_valids.append(False)
            inter_kpts_out.append(inter_kpt)
        
        return bboxes, inter_kpts_out, is_valids

    def bbox2kpts(self, bbox):
        """
        Convert from box to kpts
        
        Args:
            bbox (np.ndaray): (4,)
        
        Output:
            kpts (np.ndarray): (4, 2)
        """
        return np.array([[bbox[0], bbox[1]],   # x1, y1
                         [bbox[2], bbox[1]],   # x2, y1
                         [bbox[2], bbox[3]],   # x2, y2
                         [bbox[0], bbox[3]]])  # x1, y2

    def kpts2bbox(self, kpts):
        """
        Convert from kpts to bbox
        
        Args:
            kpts (np.ndaray): (N, 2)
        
        Output:
            bbox (np.ndarray): (4)
        """
        return np.array([kpts[:, 0].min(), kpts[:, 1].min(), kpts[:, 0].max(), kpts[:, 1].max()])

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
    is_valids = []
    kpts_list = []
    texts = []
    try:
        for line in lines:
            if len(line.split("|")) > 1:
                ann_text_infos = line.split("|")
                ann_infos = ann_text_infos[0]
                text = "|".join(ann_text_infos[1:])
                text = text.strip()
                texts.append(text)
                ann_infos = ann_infos.strip().split()
            else:  # China CTW  ##TODO: Delete this
                ann_infos = line.split()
                texts.append(ann_infos[-1])
                ann_infos = ann_infos[:-1]
            
            lbl = int(ann_infos[0])
            labels.append(lbl)

            is_valid = int(ann_infos[1])
            is_valids.append(is_valid)

            xn, yn, wn, hn = tuple(map(float, ann_infos[2:6]))
            x1 = (xn - wn / 2) * w
            y1 = (yn - hn / 2) * h
            x2 = (xn + wn / 2) * w
            y2 = (yn + hn / 2) * h
            boxes.append([x1, y1, x2, y2])

            kpts = np.array(list(map(float, ann_infos[6:]))).reshape((-1, 2))
            kpts_list.append(kpts)
        
        boxes = np.array(boxes)
        labels = np.array(labels)
        is_valids = np.array(is_valids)

        return labels, is_valids, boxes, kpts_list, texts
    
    except:
        print(ann_path)
        raise


if __name__ == "__main__":
    args = get_parser()
    
    chip_generator = ChipGenerator(args.valid_range,
                                args.c_stride,
                                args.mapping_threshold,
                                args.training_size,
                                args.n_threads,
                                args.use_neg)

    img_dirs = [args.img_train_dir, args.img_val_dir, args.img_test_dir]
    ann_dirs = [args.ann_train_dir, args.ann_val_dir, args.ann_test_dir]

    chip_generator(img_dirs, ann_dirs, args.chip_save_dir)