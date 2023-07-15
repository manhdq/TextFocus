import os
import cv2
import time
import numpy as np

import torch
import torchvision.transforms as tf

from models import get_model
from post_processing import decode_clip, decode_polys_clip
from utils import draw_bbox, draw_points, draw_mask


class PAN:
    def __init__(self, config, model_path=None, state_dict=None):
        self.device = torch.device("cuda")
        self.net = get_model(config.model)
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['state_dict'])
        elif state_dict is not None:
            self.net.load_state_dict(state_dict)

        self.num_points = 20  ##TODO: priority. Dynamic this
            
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img, short_size=736, using_rectangle=True):  ##TODO: priority. Dynamic short size
        """
        Image needs to be in RGB channel
        """
        ori_img = img.copy()
        h, w = ori_img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

        tensor = tf.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            torch.cuda.synchronize(self.device)
            start = time.time()
            preds_all = self.net(tensor)[0]
            torch.cuda.synchronize(self.device)

            if using_rectangle:
                ##TODO: Modify `decode_chip` function
                preds, points_list = decode_clip(preds_all)
            else:
                preds, points_list = decode_polys_clip(preds_all, self.num_points)  ##TODO: 

            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(points_list):
                points_list = points_list / scale
            t = time.time() - start

        return preds_all, points_list, t


class Detection:
    def __init__(self, config=None, weight_path=None):
        if config is None or weight_path is None:
            raise
        
        self.model = PAN(config, model_path=weight_path)

    def __call__(self, image, return_result=False, using_rectangle=True):
        preds, points_list, t = self.model.predict(image, using_rectangle=using_rectangle)

        if return_result:
            ##TODO: Modify `draw_bbox` func
            img = draw_points(image, points_list)
            preds = preds.detach().cpu().numpy()
            text_mask = preds[0] > 0.7311  # text  ##TODO: priority
            kernel_mask = (preds[1] > 0.7311) * text_mask  # kernel
            img = draw_mask(img, kernel_mask.astype(np.uint8), mask_color=(9, 73, 0), alpha=0.3)
            img = draw_mask(img, text_mask.astype(np.uint8), mask_color=(0, 255, 0), alpha=0.3)
            # cv2.imwrite("test_text.jpg", np.stack([text_mask]*3, axis=-1)*255)
            # cv2.imwrite("test_kernel.jpg", np.stack([kernel_mask]*3, axis=-1)*255)

        if return_result:
            return points_list, t, img
        else:
            return points_list, t