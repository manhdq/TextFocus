import time
import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU
from prediction.grid_generator import GridGenerator
from prediction.focus_chip_generator import FocusChip
from utils import visualize, visualize_focus_mask, visualize_detect


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


class PAN_FOCUS_PREDICT(nn.Module):
    def __init__(self, cfg):
        super(PAN_FOCUS_PREDICT, self).__init__()
        self.backbone = build_backbone(cfg.model.backbone)

        in_channels = cfg.model.neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(cfg.model.neck)
        self.fpem2 = build_neck(cfg.model.neck)

        self.det_head = build_head(cfg.model.detection_head)

        self.using_autofocus = cfg.using_autofocus
        assert self.using_autofocus, f"Current support using focus"
        if self.using_autofocus:
            self.focus_head = build_head(cfg.model.focus_head)
            self.grid_gen = GridGenerator(cfg.grid_generator)
            self.foc_chip_gen = FocusChip(cfg.focus_chip_generator)

        self.vis = cfg.vis
        self.cfg = cfg

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def scale_image_short(self, img):
        return scale_aligned_short(img, self.cfg.data.test.short_size)

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

    def process_pipeline(self, imgs):
        outputs = dict()

        # backbone
        f = self.backbone(imgs)
        if self.using_autofocus:
            focus_input = f[self.focus_head.focus_layer_choice]

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        # detection
        det_out = self.det_head(f)
        if self.using_autofocus:
            autofocus_out = self.focus_head(focus_input)
        
        det_out = self._upsample(det_out, imgs.size(), 4)
        autofocus_out = F.softmax(autofocus_out, dim=1)[:, 1]
        # det_res = self.det_head.get_results(det_out, img_metas, cfg)
        # outputs.update(det_res)
        outputs.update(dict(det_out=det_out, autofocus_out=autofocus_out))

        return outputs

    def forward(self, img):
        self.pred_idx = 0

        start = time.time()

        scaled_down_tiles, base_scale_down = self.grid_gen.gen_from_ori_img(img)
        if scaled_down_tiles is None:
            return {
                'bboxes': [],
                'scores': [],
                'time': 0,
                'num_tile': 1,
            }

        prediction_results = []
        num_piles = []
        for tile in scaled_down_tiles:
            final_det_out = None
            final_det_out, num_tile = self.recursive_predict(img,
                                                            tile['image'],
                                                            rank=0,
                                                            base_scale_down=base_scale_down,
                                                            last_det_out=final_det_out,
                                                            prev_left_shift=tile['prev_left_shift'],
                                                            prev_top_shift=tile['prev_top_shift'],
                                                            prev_right_shift=tile['prev_bottom_shift'],
                                                            prev_bottom_shift=tile['prev_bottom_shift'],
                                                            count_tile=0)
            prediction_results.append(final_det_out)
            num_piles.append(num_tile)
        
        final_prediction = prediction_results[0]
        if len(prediction_results) > 1:
            raise ##TODO: Code later

        img_meta = self.get_img_meta(img, img)
        img_meta = self.convert_img_meta_to_tensor(img_meta)

        outputs = self.det_head.get_results(final_prediction, img_meta, self.cfg)

        end = time.time()
        outputs.update(dict(time=end-start))
        outputs.update(dict(num_tile=int(np.sum(num_tile) + 1)))

        return outputs

    def recursive_predict(self,
                        ori_image,
                        chip,
                        rank,
                        base_scale_down,
                        last_det_out,
                        prev_left_shift,
                        prev_top_shift,
                        prev_right_shift,
                        prev_bottom_shift,
                        count_tile):
        ori_img_h, ori_img_w = ori_image.shape[:2]

        scaled_chip = self.scale_image_short(chip)
        chip_meta = self.get_img_meta(chip, scaled_chip)
        chip_scaled_size = chip_meta['org_img_size']
        chip_meta = self.convert_img_meta_to_tensor(chip_meta)

        scaled_chip = self.convert_img_to_tensor(scaled_chip).unsqueeze(0)
        scaled_chip = scaled_chip.cuda()
        data = dict(imgs=scaled_chip)
        outputs = self.process_pipeline(**data)

        det_out = outputs['det_out']
        autofocus_out = outputs['autofocus_out'][0].cpu().detach().numpy()

        if rank == 0:
            autofocus_out = np.ones_like(autofocus_out)

        to_ori_scale = base_scale_down / \
                (pow(self.cfg.autofocus.zoom_in_scale, max(rank - 1, 0)) * \
                (pow(self.cfg.autofocus.first_row_zoom_in, min(rank, 1))))
        chip_org_size = (chip_scaled_size * to_ori_scale).astype(int)
        
        final_det_out = F.interpolate(det_out, (chip_org_size[0], chip_org_size[1]), mode="nearest")
        final_det_mask = torch.ones_like(final_det_out) * rank

        # print(rank, (prev_left_shift, prev_right_shift, prev_top_shift, prev_bottom_shift))
        
        final_det_out = F.pad(final_det_out, (prev_left_shift, prev_right_shift, prev_top_shift, prev_bottom_shift), "constant", 0)
        final_det_mask = F.pad(final_det_mask, (prev_left_shift, prev_right_shift, prev_top_shift, prev_bottom_shift), "constant", 0)
        
        final_det_out = F.interpolate(final_det_out, (ori_img_h, ori_img_w), mode="nearest")
        if rank == 0:
            final_det_out = final_det_out * 0.8
        final_det_mask = F.interpolate(final_det_mask, (ori_img_h, ori_img_w), mode="nearest")

        if last_det_out is not None:
            last_det_out = (final_det_out * final_det_mask + last_det_out) / (final_det_mask + 1)
        else:
            last_det_out = final_det_out

        if rank < self.cfg.autofocus.max_focus_rank:
            chip_height, chip_width = chip.shape[:2]
            last_det_out, count_tile = self._recursive_pred_on_focus(ori_image,
                                                                autofocus_out,
                                                                chip_width,
                                                                chip_height,
                                                                rank,
                                                                base_scale_down,
                                                                last_det_out,
                                                                to_ori_scale,
                                                                prev_left_shift,
                                                                prev_top_shift,
                                                                prev_right_shift,
                                                                prev_bottom_shift,
                                                                count_tile)

        return last_det_out, count_tile

    def _recursive_pred_on_focus(self,
                                ori_image,
                                focus_mask,
                                chip_width,
                                chip_height,
                                rank,
                                base_scale_down,
                                last_det_out,
                                to_ori_scale,
                                prev_left_shift,
                                prev_top_shift,
                                prev_right_shift,
                                prev_bottom_shift,
                                count_tile):
        """
        Cut focus chip and do prediction on this chip
        """
        ori_img_h, ori_img_w = ori_image.shape[:2]
        final_det_out = last_det_out
        # Crop sub chips from the input chip by using the focus mask
        chip_coords = self.foc_chip_gen(focus_mask, chip_width, chip_height)
        for chip_coord in chip_coords:
            for tile_coord in self.grid_gen.gen_from_chip(chip_coord, rank):
                # Convert chip coordinates to original coordinates by
                # scaling chip coordinates
                # and shift chip coordinates to match the top-left of the original image
                resized_bbox = self.batch_scale_n_shift_dets(py_preds=[np.expand_dims(tile_coord, axis=0).astype(float)],
                                                scale=to_ori_scale,
                                                left_shift=prev_left_shift,
                                                top_shift=prev_top_shift,
                                                right_shift=prev_right_shift,
                                                bottom_shift=prev_bottom_shift)[0]
                ori_x1, ori_y1, ori_x2, ori_y2 = resized_bbox[0]

                # Crop interested region on original image
                ori_chip_crop = ori_image[round(ori_y1):round(ori_y2),
                                        round(ori_x1):round(ori_x2), :]

                zoom_scale = self.cfg.autofocus.zoom_in_scale if rank > 0 else self.cfg.autofocus.first_row_zoom_in
                zoom_in_x1, zoom_in_y1, zoom_in_x2, zoom_in_y2 = \
                    list(map(lambda x: x * zoom_scale, tile_coord))
                zoom_in_chip_w = round(zoom_in_x2 - zoom_in_x1)
                zoom_in_chip_h = round(zoom_in_y2 - zoom_in_y1)

                zoom_in_chip_crop = cv2.resize(ori_chip_crop,
                                            (zoom_in_chip_w, zoom_in_chip_h),
                                            interpolation=self.grid_gen.interpolation)

                final_det_out, cur_count_tile = self.recursive_predict(ori_image,
                                                                    zoom_in_chip_crop,
                                                                    rank + 1,
                                                                    base_scale_down,
                                                                    final_det_out,
                                                                    int(ori_x1),
                                                                    int(ori_y1),
                                                                    int(ori_img_w - ori_x2),
                                                                    int(ori_img_h - ori_y2),
                                                                    0)
                count_tile = count_tile + cur_count_tile + 1
        return final_det_out, count_tile

    def batch_scale_n_shift_dets(self, py_preds, scale, left_shift, top_shift, right_shift, bottom_shift):
        '''
        Scale and shift old coordinate to new coordinate
        `py_preds` contains list of lm preds according to `GDN` level
        '''
        _py_preds = []
        for cur_py_preds in py_preds:
            _cur_py_preds = cur_py_preds.copy()
            _cur_py_preds *= scale
            
            _cur_py_preds[..., 0::2] = _cur_py_preds[..., 0::2] + left_shift
            _cur_py_preds[..., 1::2] = _cur_py_preds[..., 1::2] + top_shift
            
            _py_preds.append(_cur_py_preds)
        return _py_preds

    def focus(self, imgs):
        ##NOTE: This function only use for inference
        assert self.using_autofocus

        outputs = dict()
        
        # backbone
        f = self.backbone(imgs)
        if self.using_autofocus:
            focus_input = f[self.focus_head.focus_layer_choice]

        autofocus_out = self.focus_head(focus_input)
        autofocus_out = F.softmax(autofocus_out, dim=1)[:, 1]

        outputs.update(dict(
            backbone_out = f,
            autofocus_out=autofocus_out
        ))

        return outputs

    def get_det_map_after_backbone(self, backbone_outs):
        # reduce channel
        f1 = self.reduce_layer1(backbone_outs[0])
        f2 = self.reduce_layer2(backbone_outs[1])
        f3 = self.reduce_layer3(backbone_outs[2])
        f4 = self.reduce_layer4(backbone_outs[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        # detection
        det_out = self.det_head(f)
        return det_out