import os
import cv2
import math
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from .data_utils import image_label
from .augment import DataAugment

data_aug = DataAugment()


class ImageDataset(Dataset):
    def __init__(self, data_list: list, input_size: int, img_channel: int, shrink_ratio: float, transform=None,
                 focus_gen=None, max_points=20, train=True):

        self.max_points = max_points
        self.data_list = self.load_data(data_list)
        self.train = train
        self.input_size = input_size
        self.img_channel = img_channel
        self.transform = transform
        self.shrink_ratio = shrink_ratio
        self.focus_gen = focus_gen

    def __getitem__(self, index):
        img_path, text_polys, text_tags, text_lengths = self.data_list[index]
        im = cv2.imread(img_path, 1 if self.img_channel == 3 else 0)

        if self.img_channel == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        img, score_maps, training_mask, text_polys = image_label(im, text_polys, text_tags, text_lengths, self.input_size,
                                                                self.shrink_ratio, degrees=90, train=self.train)

        if self.train:
            ##TODO: Modify `Random Resize Crop` func
            imgs, text_polys = data_aug.random_crop([img, score_maps.transpose((1, 2, 0)), training_mask],
                                        text_polys, text_tags,
                                        (self.input_size, self.input_size))
            img = imgs[0]
            score_maps = imgs[1].transpose((2, 0, 1))
            training_mask = imgs[2]

        # Get autofocus_mask
        h, w, _ = img.shape
        focus_mask, flattened_focus_mask = None, None
        if self.focus_gen is not None:
            focus_mask, flattened_focus_mask = self._prepare_focus_mask_by_landmarks((w, h), text_polys, text_tags)
        
            # show_img = self.visualize_gt(img, text_polys, focus_mask)
            # cv2.imwrite("test_mask.jpg", show_img)
        else:
            ##TODO: Simplify this
            focus_mask = np.zeros((1))
            flattened_focus_mask = np.zeros((1))

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        # cv2.imwrite("test.jpg", np.stack([score_maps[0]]*3, -1).astype(np.uint8)*255)
        # cv2.imwrite("test_1.jpg", np.stack([score_maps[1]]*3, -1).astype(np.uint8)*255)
        # exit()
        return img, score_maps, training_mask, focus_mask, flattened_focus_mask

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

        # for contour in contours:
        #     box = contour.box.astype(int)
        #     image_show = cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]),
        #                             (255, 0, 0), 1)

        image_show = cv2.polylines(image_show,
                                [points.astype(int) for points in contours], True, (0, 0, 255), 2)
        image_show = cv2.polylines(image_show,
                                [points.astype(int) for points in contours], True, (0, 255, 0), 2)

        show_gt = cv2.resize(image_show, (320, 320))

        return show_gt

    def _prepare_focus_mask_by_landmarks(self, img_size, lms_group, text_tags):
        """
        Prepare focus_mask corresponding with new image size"""
        w, h = img_size
        mask_w = math.ceil(w / self.focus_gen.stride)
        mask_h = math.ceil(h / self.focus_gen.stride)
        mask = np.zeros((mask_h, mask_w)).astype(int)

        mask = self.focus_gen.calculate_mask_by_landmarks_group(lms_group, text_tags, mask)
        flattened_mask = mask.reshape(mask.shape[0] * mask.shape[1])
        return mask, flattened_mask

    def load_data(self, data_list: list) -> list:
        t_data_list = []
        for img_path, label_path in data_list:
            bboxs, text_tags, text_lengths = self._get_annotation(label_path)
            if len(bboxs) > 0:
                t_data_list.append((img_path, bboxs, text_tags, text_lengths))
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_image(self, index: int) -> str:
        image_info = self.coco.loadImgs(index)[0]
        path = os.path.join(self.root_dir, image_info['file_name'])
        return path

    def _get_annotation(self, label_path: str) -> tuple:
        boxes = []
        text_tags = []
        text_lengths = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                ann_infos = line.split(" | ")
                text = ann_infos[1:]
                text = " | ".join(text).strip()
                ann_infos = ann_infos[0].strip().split()
                is_valid = int(ann_infos[1])
                gt = list(map(float, ann_infos[6:]))
                assert len(gt) % 2 == 0
                # try:
                box = np.array(gt).reshape(-1, 2).astype(int)
                assert box.shape[0] <= self.max_points, box.shape
                text_lengths.append(box.shape[0])
                if box.shape[0] < self.max_points:
                    box_pad = np.stack([box[-1]]*(self.max_points - box.shape[0]))
                    box = np.concatenate([box, box_pad], 0)
                
                # print(box.shape)
                # box = order_points_clockwise(np.array(gt).reshape(-1, 2))
                ##TODO: Fix this later
                if cv2.arcLength(box, True) > 0:
                    boxes.append(box)
                    if is_valid == 0:
                        text_tags.append(False)
                    else:
                        text_tags.append(True)
                # except:
                #     print('load label failed on {}'.format(label_path))
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=bool), np.array(text_lengths, dtype=int)

    def __len__(self):
        return len(self.data_list)
