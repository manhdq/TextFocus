# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfglib.config import config as cfg
from network.layers.model_block import FPN
from network.layers.transformer import Transformer
from network.layers.autofocus import AutoFocus
from network.layers.gcn_utils import get_node_feature
from utils.misc import get_sample_point


class Evolution(nn.Module):
    def __init__(self, node_num, adj_num, is_training=True, device=None, model="snake"):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.adj_num = adj_num
        self.device = device
        self.is_training = is_training
        self.clip_dis = 16  ##TODO: What is it?

        ##TODO: Dynamic param `iter`
        self.iter = 3
        if model == "gcn":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = GCN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "BT":
            self.adj = None
            for i in range(self.iter):
                ##TODO: Dynamic these params
                evolve_gcn = Transformer(36, 128, num_heads=8,
                                        dim_feedforward=1024, drop_rate=0.0,
                                        if_resi=True, block_nums=3)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        else:
            self.adj = get_adj_ind(self.adj_num, self.node_num, self.device)
            for i in range(self.iter):
                evolve_gcn = DeepSnake(state_dim=128, feature_dim=36, conv_type='dgrid')
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):
        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
        else:  ##TODO: This func recreate sample point for proposal, do we need modify it for cleaner?
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold  ##TODO: What this threshold??
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)
        
        return init_polys, inds, None

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2) * cfg.scale, cnn_feature.size(3) * cfg.scale
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        i_poly = i_it_poly + torch.clamp(snake(node_feats, self.adj).permute(0, 2, 1), -self.clip_dis, self.clip_dis)
        if self.is_training:
            ##TODO: Why not clip according to `h` during training
            i_poly = torch.clamp(i_poly, 0, w-1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)
        return i_poly

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        if self.is_training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter+1)], inds, confidences

        py_preds = [init_polys, ]
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            init_polys = self.evolve_poly(evolve_gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)

        return py_preds, inds, confidences


class TextBPNPlusPlusNet(nn.Module):

    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        ##TODO: Modify this `is_training` default
        self.fpn = FPN(self.backbone_name, is_training=(not cfg.resume and is_training))

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )
        ##TODO: Modify for dynamic params
        self.BPN = Evolution(cfg.num_points, adj_num=4,
                            is_training=is_training, device=cfg.device, model="BT")

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(cfg.device))
        self.load_state_dict(state_dict['model'], strict=(not self.is_training))

    def forward(self, input_dict, test_speed=False):
        output = {}
        b, c, h, w = input_dict["img"].shape
        if self.is_training or cfg.exp_name in ['ArT', 'MLT2017', "MLT2019"] or test_speed:
            image = input_dict["img"]
        else:
            image = torch.zeros((b, c, cfg.test_size[1], cfg.test_size[0]), dtype=torch.float32).to(cfg.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]

        up1, _, _, _, _ = self.fpn(image)
        up1 = up1[:, :, :h // cfg.scale, :w // cfg.scale]

        preds = self.seg_head(up1)
        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)
        cnn_feats = torch.cat([up1, fy_preds], dim=1)

        py_preds, inds, confidences = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")

        output['fy_preds'] = fy_preds
        output['py_preds'] = py_preds
        output['inds'] = inds
        output['confidences'] = confidences

        return output


##TODO: Add `load_model` func
class TextBPNFocus(nn.Module):
    """
    The implementation of the Text model with Autofocus ability.
    The code will be while the same as `TextBPN++` but with adding AutoFocus func.
    
    TextBPNFocus = TextBPN++ + AutoFocus + Magic
    """
    def __init__(self, backbone='vgg', is_training=True,
                using_autofocus=True, focus_layer_choice=2):
        super().__init__()

        self.TextBPN = TextBPNPlusPlusNet(backbone, is_training)
        self.is_training = is_training

        self.using_autofocus = using_autofocus

        if self.using_autofocus:
            ##TODO: Make this option param dynamic
            # 0: up1
            # 1: up2
            # 2: up3
            # 3: up4
            # 4: up5
            self.focus_layer_choice = focus_layer_choice
            autofocus_in_channels = [
                32, 32, 64, 128, 256
            ][self.focus_layer_choice]
            self.autofocus = AutoFocus(autofocus_in_channels)

    def forward(self, input_dict, test_speed=False):
        output = {}
        b, c, h, w = input_dict["img"].shape
        if self.is_training or cfg.exp_name in ['ArT', 'MLT2017', "MLT2019"] or test_speed:
            image = input_dict["img"]
        else:
            image = torch.zeros((b, c, cfg.test_size[1], cfg.test_size[0]), dtype=torch.float32).to(cfg.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]

        up1, up2, up3, up4, up5 = self.TextBPN.fpn(image)
        # print(up1.shape)
        # print(up2.shape)
        # print(up3.shape)
        # print(up4.shape)
        # print(up5.shape)
        # exit()
        up1 = up1[:, :, :h // cfg.scale, :w // cfg.scale]

        preds = self.TextBPN.seg_head(up1)
        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)
        cnn_feats = torch.cat([up1, fy_preds], dim=1)

        py_preds, inds, confidences = self.TextBPN.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")

        output['fy_preds'] = fy_preds
        output['py_preds'] = py_preds
        output['inds'] = inds
        output['confidences'] = confidences

        if self.using_autofocus:
            # Feature map extraction for Autofocus phase
            # Get scale
            if self.focus_layer_choice < 3:  # up1, up2, up3
                focus_scale = cfg.scale  # 4
            elif self.focus_layer_choice == 3:  # up4
                focus_scale = cfg.scale * 2  # 8
            elif self.focus_layer_choice == 4:  # up5
                focus_scale = cfg.scale * 4  # 16
            else:
                raise

            focus_layer_feat = [
                up1, up2, up3, up4, up5
            ][self.focus_layer_choice]
            autofocus_out = self.autofocus(focus_layer_feat)
            if self.is_training:
                autofocus_out = torch.reshape(autofocus_out,
                                        shape=(autofocus_out.shape[0], 2, -1))
            else:
                autofocus_out = F.softmax(autofocus_out, dim=1)
            output['autofocus_preds'] = autofocus_out

        return output