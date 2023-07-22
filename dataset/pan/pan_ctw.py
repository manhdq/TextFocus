import random
import os
import cv2
import math
import mmcv
import numpy as np
import Polygon as plg
import pyclipper
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

"""
Format data

    |Images
    |    |train 
    |    |test
    |gt
    |    |train 
    |    |test
"""

def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print(img_path)
        raise
    return img

# # Function get annotation (original version)
# def get_ann(img, gt_path):
#     h, w = img.shape[0:2]
#     lines = mmcv.list_from_file(gt_path)
#     bboxes = []
#     words = []
#     for line in lines:
#         line = line.replace('\xef\xbb\xbf', '')
#         gt = line.split(',')

#         x1 = np.int_(gt[0])
#         y1 = np.int_(gt[1])

#         bbox = [np.int_(gt[i]) for i in range(4, 32)]
#         bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
#         bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

#         bboxes.append(bbox)
#         words.append('???')
#     return bboxes, words


def get_ann(img, gt_path):
    h, w = img.shape[0: 2]

    ptses = []
    labels = []

    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data_line = line.split('|')
            label = data_line[1].strip()
            data = data_line[0].strip().split(' ')
            xywh_yolo = data[2:6]
            pts = data[6:]
            pts = [int(float(item)) for item in pts]
            pair = int(len(pts) / 2)
            pts = np.asarray(pts) / ([w * 1.0, h * 1.0] * pair)
            ptses.append(pts)
            labels.append(label)
    return ptses, labels


def random_horizontal_flip(imgs, bboxes):
    h, w = imgs[0].shape[0:2]
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
        for i in range(len(bboxes)):
            bboxes[i][:, 0] = w - bboxes[i][:, 0]
    return imgs, bboxes


def random_rotate(imgs, bboxes):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        h, w = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img_rotation = cv2.warpAffine(img,
                                      rotation_matrix, (w, h),
                                      flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    ##kpts
    h, w = imgs[0].shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    for i in range(len(bboxes)):
        bbox = bboxes[i].astype(np.float32)
        bbox_homogeneous = np.column_stack((bbox, np.ones(len(bbox))))
        rotated_bbox = bbox_homogeneous.dot(rotation_matrix.T)
        bboxes[i] = rotated_bbox[:, :2]
    return imgs, bboxes


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=640):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
    return img


def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_crop_padding(imgs, target_size, bboxes):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs, bboxes

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0
                                                   for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0, ))
        n_imgs.append(img_p)
    for idx in range(len(bboxes)):
        bboxes[idx][:, 0] = bboxes[idx][:, 0] - j
        bboxes[idx][:, 1] = bboxes[idx][:, 1] - i
    return n_imgs, bboxes


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5),
                         max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


class PAN_CTW(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=640,
                 kernel_scale=0.7,
                 read_type='pil',
                 report_speed=False,
                 focus_gen=None,
                 root_dir=None,
                 train_data_dir=None,
                 test_data_dir=None,
                 train_gt_dir=None,
                 test_gt_dir=None):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (
            img_size is None or isinstance(img_size, tuple)) else (img_size,
                                                                   img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.read_type = read_type

        if split == 'train':
            data_dirs = [os.path.join(root_dir, train_data_dir)]
            gt_dirs = [os.path.join(root_dir, train_gt_dir)]
        elif split == 'test':
            data_dirs = [os.path.join(root_dir, test_data_dir)]
            gt_dirs = [os.path.join(root_dir, test_gt_dir)]
        else:
            print('Error: split must be train or test!')
            raise

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [
                img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')
            ]
            img_names.extend([
                img_name for img_name in mmcv.utils.scandir(data_dir, '.png')
            ])

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

        # ##DEBUG
        # self.img_paths = self.img_paths[0:20]
        # self.gt_paths = self.gt_paths[5:6]
        
        if report_speed:
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

        self.max_word_num = 200

        self.focus_gen = focus_gen

    def __len__(self):
        return len(self.img_paths)

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
                                [points.astype(int) for points in contours], True, (0, 0, 255), 1)
        image_show = cv2.polylines(image_show,
                                [points.astype(int) for points in contours], True, (0, 255, 0), 1)

        show_gt = cv2.resize(image_show, (320, 320))

        return show_gt

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann(img, gt_path)

        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]

        if self.is_transform:
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(
                    bboxes[i] * ([img.shape[1], img.shape[0]] *
                                 (bboxes[i].shape[0] // 2)),
                    (bboxes[i].shape[0] // 2, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            imgs, bboxes = random_horizontal_flip(imgs, bboxes)
            imgs, bboxes = random_rotate(imgs, bboxes)
            imgs, bboxes = random_crop_padding(imgs, self.img_size, bboxes)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[
                1], imgs[2], imgs[3:]

        # Get autofocus mask
        h, w, _ = img.shape
        focus_mask, flattened_focus_mask = None, None
        # cv2.imwrite("test.jpg", (np.stack([gt_kernels[0]]*3, -1)) * 255)
        if self.focus_gen is not None:
            text_tags = [True] * len(bboxes)
            focus_mask, flattened_focus_mask = self._prepare_focus_mask_by_landmarks((w, h), bboxes, text_tags)

            # print(focus_mask.shape)
            # show_img = self.visualize_gt(img, bboxes, focus_mask)
            # cv2.imwrite("test_mask.jpg", show_img)
        else:
            ##TODO: Simplify this
            focus_mask = np.zeros((1))
            flattened_focus_mask = np.zeros((1))

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255,
                                         saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        focus_mask = torch.from_numpy(focus_mask).long()
        flattened_focus_mask = torch.from_numpy(flattened_focus_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
            focus_mask=focus_mask,
            flattened_focus_mask=flattened_focus_mask
        )

        return data

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)
        img_meta = dict(org_img_size=np.array(img.shape[:2]))

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(img_size=np.array(img.shape[:2])))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        data = dict(imgs=img, img_metas=img_meta)

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)

    def get_image(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)

        return img

    def scale_image_short(self, img):
        return scale_aligned_short(img, self.short_size)

    def get_img_meta(self, img, scaled_img):
        img_meta = dict(org_img_size=np.array(img.shape[:2]))
        img_meta.update(dict(img_size=np.array(scaled_img.shape[:2])))

        return img_meta

    def convert_img_meta_to_tensor(self, img_meta):
        for k, v in img_meta.items():
            img_meta[k] = torch.from_numpy(v[None])
        return img_meta

    def convert_img_to_tensor(self, img):
        assert isinstance(img, np.ndarray), type(img)

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        return img