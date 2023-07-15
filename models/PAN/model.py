# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun

import torch
from torch import nn
import torch.nn.functional as F


from .modules import *
from .autofocus import AutoFocus

backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
                 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
                 'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
                 'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
                 'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
                 'resnext50_32x4d': {'models': resnext50_32x4d, 'out': [256, 512, 1024, 2048]},
                 'resnext101_32x8d': {'models': resnext101_32x8d, 'out': [256, 512, 1024, 2048]},
                 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}
                 }

segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}


# 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
# 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
# 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}}


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = segmentation_head_dict[segmentation_head](backbone_out, **model_config)
        self.name = '{}_{}'.format(backbone, segmentation_head)

        self.is_training = model_config['is_training']
        self.using_autofocus = model_config['using_autofocus']
        self.focus_layer_choice = model_config['focus_layer_choice']

        if self.using_autofocus:
            autofocus_in_channels = [
                64, 128, 256, 512
            ][self.focus_layer_choice]
            self.autofocus = AutoFocus(autofocus_in_channels)

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_head_out = self.segmentation_head(backbone_out)
        y = F.interpolate(segmentation_head_out, size=(H, W), mode='bilinear', align_corners=True)

        if self.using_autofocus:
            # Feature map extraction for Autofocus phase
            focus_layer_feat = backbone_out[self.focus_layer_choice]
            autofocus_out = self.autofocus(focus_layer_feat)
            if self.is_training:
                autofocus_out = torch.reshape(autofocus_out,
                                        shape=(autofocus_out.shape[0], 2, -1))
            else:
                autofocus_out = F.softmax(autofocus_out, dim=1)
            
            return y, autofocus_out
        return y


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model_config = {
        'backbone': 'shufflenetv2',
        'fpem_repeat': 4,  # fpem模块重复的次数
        'pretrained': True,  # backbone 是否使用imagesnet的预训练模型
        'result_num': 7,
        'segmentation_head': 'FPEM_FFM'  # 分割头，FPN or FPEM_FFM
    }
    model = Model(model_config=model_config).to(device)
    y = model(x)
    print(y.shape)
    # print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
