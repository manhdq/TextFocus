import copy
import cv2
import math
import numpy as np
from PIL import Image
from scipy import ndimage as ndimg

import torch

from cfglib.config import config as cfg
from utils.misc import get_sample_point, split_edge_sequence


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
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
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

    def get_sample_point(self, size=None):
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [self.points.astype(np.int32)], color=(1,))
        control_points = get_sample_point(mask, cfg.num_points, cfg.approx_factor)

        return control_points

    def __repr__(self,):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):
    def __init__(self, transform, focus_gen=None, is_training=False):
        super().__init__()
        self.transform = transform
        self.focus_gen = focus_gen
        self.is_training = is_training
        ##TODO: Modify this??
        self.min_text_size = 4
        self.jitter = 0.65  # Random adjust proposal points for better training
        self.th_b = 0.35  # distance threshold for proposal points selecting from gt boundary

    @staticmethod
    def sigmoid_alpha(x, k):
        betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
        dm = max(np.max(x), 0.0001)
        res = (2 / (1 + np.exp(-x * k / dm)) - 1) * betak
        return np.maximum(0, res)

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        Fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))

    @staticmethod
    def generate_proposal_point(
        test_mask, num_points, approx_factor, jitter=0.0, distance=10.0
    ):
        # Get proposal point in contours
        h, w = test_mask.shape[:2]
        contours, _ = cv2.findContours(
            test_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        epsilon = approx_factor * cv2.arcLength(contours[0], True)
        ##TODO: Why we need this?? why we need less vertices
        approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
        ctrl_points = split_edge_sequence(approx, num_points)

        ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

        if jitter > 0:
            x_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance * jitter
            y_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance * jitter
            ctrl_points[:, 0] += x_offset.astype(np.int32)
            ctrl_points[:, 1] += y_offset.astype(np.int32)
        ctrl_points[:, 0] = np.clip(ctrl_points[:, 0], 1, w - 2)
        ctrl_points[:, 1] = np.clip(ctrl_points[:, 1], 1, h - 2)
        return ctrl_points

    @staticmethod
    def compute_direction_field(inst_mask, h, w):
        _, labels = cv2.distanceTransformWithLabels(
            inst_mask,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        # Compute the direction field
        index = np.copy(labels)
        index[inst_mask > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y
        grid = np.indices(inst_mask.shape)
        grid = grid.astype(float)
        diff = nearPixel - grid

        return diff

    ##TODO:
    def visualize_gt(self, image, contours, focus_mask):
    
        image_show = image.copy()
        h, w = image_show.shape[:2]
        image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min()) * 255
        image_show = np.ascontiguousarray(image_show[:, :, ::-1]).astype(np.uint8)

        # Add focus mask
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

        image_show = cv2.polylines(image_show,
                                [contour.points.astype(np.int) for contour in contours], True, (0, 0, 255), 2)
        image_show = cv2.polylines(image_show,
                                [contour.points.astype(np.int) for contour in contours], True, (0, 255, 0), 2)

        show_gt = cv2.resize(image_show, (320, 320))

        return show_gt

    def _norm_annotation(self, boxes, lms, img_size):
        """
        Scale bbox and lm with new image scale
        """
        w, h = img_size
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        # Normalize lm
        if lms is not None:
            lms[:, 0::2] /= w
            lms[:, 1::2] /= h
        return boxes, lms
    
    ##TODO: Delete this?
    def _prepare_focus_mask_by_boxes(self, img_size, boxes):
        """
        Prepare focus mask corresponding with new image size
        """
        ##TODO: adapt with `scale` in model
        w, h = img_size
        mask_w = math.ceil(w / self.focus_gen.stride)
        mask_h = math.ceil(h / self.focus_gen.stride)
        mask = np.zeros((mask_h, mask_w)).astype(np.long)
        ##TODO: should we adapt this with lms??
        for bb in boxes:
            scaled_bb = bb * ([w, h] * 2)
            mask = self.focus_gen.calculate_mask(
                scaled_bb[0], scaled_bb[1], scaled_bb[2], scaled_bb[3], mask
            )
        flattened_mask = mask.reshape(mask.shape[0] * mask.shape[1])
        return mask, flattened_mask

    def _prepare_focus_mask_by_landmarks(self, img_size, lms_group):
        """
        Prepare focus_mask corresponding with new image size"""
        w, h = img_size
        mask_w = math.ceil(w / self.focus_gen.stride)
        mask_h = math.ceil(h / self.focus_gen.stride)
        mask = np.zeros((mask_h, mask_w)).astype(np.long)

        mask = self.focus_gen.calculate_mask_by_landmarks_group(lms_group, mask)
        flattened_mask = mask.reshape(mask.shape[0] * mask.shape[1])
        return mask, flattened_mask

    def make_text_region(self, img, polygons):
        h, w = img.shape[0], img.shape[1]
        mask_zeros = np.zeros(img.shape[:2], np.uint8)

        train_mask = np.ones((h, w), np.uint8)
        tr_mask = np.zeros((h, w), np.uint8)
        weight_matrix = np.zeros((h, w), dtype=np.float)
        direction_field = np.zeros((2, h, w), dtype=np.float)
        distance_field = np.zeros((h, w), np.float)

        gt_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float)
        proposal_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float)
        ignore_tags = np.zeros((cfg.max_annotation,), dtype=np.int)

        if polygons is None:
            return (
                train_mask,
                tr_mask,
                distance_field,
                direction_field,
                weight_matrix,
                gt_points,
                proposal_points,
                ignore_tags,
            )

        ##TODO: sort the polygons by decensding area
        for idx, polygon in enumerate(polygons):
            if idx >= cfg.max_annotation:
                break
            polygon.points[:, 0] = np.clip(polygon.points[:, 0], 1, w - 2)
            polygon.points[:, 1] = np.clip(polygon.points[:, 1], 1, h - 2)
            gt_points[idx, :, :] = polygon.get_sample_point(size=(h, w))
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int)], color=(idx + 1,))

            inst_mask = mask_zeros.copy()
            cv2.fillPoly(inst_mask, [polygon.points.astype(np.int32)], color=(1,))
            dmp = ndimg.distance_transform_edt(inst_mask)  # distance transform

            if (
                polygon.text == "#"
                or np.max(dmp) < self.min_text_size
                or np.sum(inst_mask) < 150  ##TODO: should we change this??
            ):
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
                ignore_tags[idx] = -1
            else:
                ignore_tags[idx] = 1

            proposal_points[idx, :, :] = self.generate_proposal_point(
                dmp / (np.max(dmp) + 1e-3) >= self.th_b,
                cfg.num_points,
                cfg.approx_factor,
                jitter=self.jitter,
                distance=self.th_b * np.max(dmp),
            )

            distance_field[:, :] = np.maximum(
                distance_field[:, :], dmp / (np.max(dmp) + 1e-3)
            )

            ##TODO: Do we need this to weight small text
            weight_matrix[inst_mask > 0] = 1.0 / np.sqrt(inst_mask.sum())
            diff = self.compute_direction_field(inst_mask, h, w)
            direction_field[:, inst_mask > 0] = diff[:, inst_mask > 0]

        ##### Background #####
        weight_matrix[tr_mask == 0] = 1.0 / np.sqrt(np.sum(tr_mask == 0))
        # diff = self.compute_direction_field((tr_mask == 0).astype(np.uint8), h, w)
        # direction_field[:, tr_mask == 0] = diff[:, tr_mask == 0]

        # Get autofocus annotations
        focus_mask, flattened_focus_mask = self._prepare_focus_mask_by_landmarks((w, h), gt_points[ignore_tags==1])
        ##TODO:
        # show_img = self.visualize_gt(img, polygons, focus_mask)
        # cv2.imwrite("test_mask.jpg", show_img)
        # exit()

        train_mask = np.clip(train_mask, 0, 1)

        return (
            train_mask,
            tr_mask,
            distance_field,
            direction_field,
            weight_matrix,
            gt_points,
            proposal_points,
            ignore_tags,
            # Autofocus
            focus_mask,
            flattened_focus_mask,
        )

    def get_training_data(self, image, polygons, image_id=None, image_path=None):
        np.random.seed()

        if self.transform:
            image, polygons = self.transform(
                copy.deepcopy(image), copy.deepcopy(polygons)
            )
            ##TODO: Get box for polygons, modify in `transform` later
            # Get box for polygons
            for polygon in polygons:
                polygon.box = points2box(polygon.points)

        train_mask, tr_mask, distance_field, direction_field, \
            weight_matrix, gt_points, proposal_points, ignore_tags, \
            focus_mask, flattened_focus_mask = \
                self.make_text_region(image, polygons)

        # To pytorch channel sequence
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        train_mask = torch.from_numpy(train_mask).bool()
        tr_mask = torch.from_numpy(tr_mask).int()
        distance_field = torch.from_numpy(distance_field).float()
        direction_field = torch.from_numpy(direction_field).float()
        weight_matrix = torch.from_numpy(weight_matrix).float()
        gt_points = torch.from_numpy(gt_points).float()
        proposal_points = torch.from_numpy(proposal_points).float()
        ignore_tags = torch.from_numpy(ignore_tags).int()

        focus_mask = torch.from_numpy(focus_mask).long()
        flattened_focus_mask = torch.from_numpy(flattened_focus_mask).long()

        return (
            image,
            train_mask,
            tr_mask,
            distance_field,
            direction_field,
            weight_matrix,
            gt_points,
            proposal_points,
            ignore_tags,
            # Autofocus
            focus_mask,
            flattened_focus_mask,
        )
    
    ##TODO: 
    def get_test_data(self, image, polygons, image_id=None, image_path=None):
        if self.transform:
            (image, polygons), _ = self.transform(
                copy.deepcopy(image), copy.deepcopy(polygons), return_pads=False
            )
            ##TODO: Get box for polygons, modify in `transform` later
            # Get box for polygons
            for polygon in polygons:
                polygon.box = points2box(polygon.points)

        train_mask, tr_mask, distance_field, direction_field, \
            weight_matrix, gt_points, proposal_points, ignore_tags, \
            focus_mask, flattened_focus_mask = \
                self.make_text_region(image, polygons)

        # To pytorch channel sequence
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        train_mask = torch.from_numpy(train_mask).bool()
        tr_mask = torch.from_numpy(tr_mask).int()
        distance_field = torch.from_numpy(distance_field).float()
        direction_field = torch.from_numpy(direction_field).float()
        weight_matrix = torch.from_numpy(weight_matrix).float()
        gt_points = torch.from_numpy(gt_points).float()
        proposal_points = torch.from_numpy(proposal_points).float()
        ignore_tags = torch.from_numpy(ignore_tags).int()

        focus_mask = torch.from_numpy(focus_mask).long()
        flattened_focus_mask = torch.from_numpy(flattened_focus_mask).long()

        return (
            image,
            train_mask,
            tr_mask,
            distance_field,
            direction_field,
            weight_matrix,
            gt_points,
            proposal_points,
            ignore_tags,
            # Autofocus
            focus_mask,
            flattened_focus_mask,
        )

    # ##TODO: Code consistance for training and test data
    # def get_test_data(self, image, polygons=None, image_id=None, image_path=None):
    #     H, W, _ = image.shape
    #     if self.transform:
    #         image, polygons = self.transform(image, polygons)

    #     # Max point per polygon for annotation
    #     points = np.zeros((cfg.max_annotation, cfg.num_points, 2))
    #     length = np.zeros(cfg.max_annotation, dtype=int)
    #     label_tag = np.zeros(cfg.max_annotation, dtype=int)
    #     if polygons is not None:
    #         for i, polygon in enumerate(polygons):
    #             pts = polygon.points
    #             points[i, :pts.shape[0]] = polygon.points
    #             length[i] = pts.shape[0]
    #             if polygon.text != "#":
    #                 label_tag[i] = 1
    #             else:
    #                 label_tag[i] = -1

    #     meta = {
    #         "image_id": image_id,
    #         "image_path": image_path,
    #         "annotation": points,
    #         "n_annotaiton": length,
    #         "label_tag": label_tag,
    #         "Height": H,
    #         "Width": W,
    #     }

    #     # To pytorch channel sequence
    #     image = image.transpose(2, 0, 1)

    #     return image, meta

    def _len__(self):
        raise NotImplementedError