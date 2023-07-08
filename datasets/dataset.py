import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data_utils import image_label
from utils import order_points_clockwise
from pycocotools.coco import COCO

class ImageDataset(Dataset):
    def __init__(self, data_list: list, input_size: int, img_channel: int, shrink_ratio: float, transform=None,
                 target_transform=None, train=True):

        self.data_list = self.load_data(data_list)
        self.train = train
        self.input_size = input_size
        self.img_channel = img_channel
        self.transform = transform
        self.target_transform = target_transform
        self.shrink_ratio = shrink_ratio

    def __getitem__(self, index):
        img_path, text_polys, text_tags = self.data_list[index]
        im = cv2.imread(img_path, 1 if self.img_channel == 3 else 0)
        if self.img_channel == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img, score_map, training_mask = image_label(im, text_polys, text_tags, self.input_size,
                                                    self.shrink_ratio, degrees=90, train=self.train)
        # img = draw_bbox(img,text_polys)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_map = self.target_transform(score_map)
            training_mask = self.target_transform(training_mask)
        return img, score_map, training_mask

    def load_data(self, data_list: list) -> list:
        t_data_list = []
        for img_path, label_path in data_list:
            bboxs, text_tags = self._get_annotation(label_path)
            if len(bboxs) > 0:
                t_data_list.append((img_path, bboxs, text_tags))
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
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)
