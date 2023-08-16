import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU


class PAN(nn.Module):
    def __init__(self, backbone, neck, detection_head, using_autofocus=False, focus_head=None):
        super(PAN, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)

        self.using_autofocus = using_autofocus
        if using_autofocus:
            self.focus_head = build_head(focus_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                focus_mask=None,
                flattened_focus_mask=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)
        if self.using_autofocus:
            focus_input = f[self.focus_head.focus_layer_choice]

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

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

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)
        if self.using_autofocus:
            autofocus_out = self.focus_head(focus_input)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels,
                                          training_masks, gt_instances,
                                          gt_bboxes)
            outputs.update(det_loss)

            if self.using_autofocus:
                focus_loss = self.focus_head.loss(autofocus_out, focus_mask, flattened_focus_mask)
                outputs.update(focus_loss)
        else:
            if not self.using_autofocus:
                det_out = self._upsample(det_out, imgs.size(), 4)
                det_res = self.det_head.get_results(det_out, img_metas, cfg)
                outputs.update(det_res)

            else:
                det_out = self._upsample(det_out, imgs.size(), 4)
                autofocus_out = F.softmax(autofocus_out, dim=1)[:, 1]
                # det_res = self.det_head.get_results(det_out, img_metas, cfg)
                # outputs.update(det_res)
                outputs.update(dict(det_out=det_out, autofocus_out=autofocus_out))

        return outputs

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