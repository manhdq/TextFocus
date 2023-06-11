# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfglib.config import config as cfg
from network.reg_loss import PolyMatchingLoss
from network.focal_loss import FocalLoss


class TextLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.PolyMatchingLoss = PolyMatchingLoss(cfg.num_points, cfg.device)
        self.KL_loss = torch.nn.KLDivLoss(reduce=False, size_average=False)

        # Focus loss
        self.focus_loss = FocalLoss(gamma=cfg.focal_gamma, ignore_index=-1)

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0  ##TODO: ???
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < eps)], 3*positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss / batch_size

    @staticmethod
    def loss_calc_flux(pred_flux, gt_flux, weight_matrix, mask, train_mask):

        # norm loss
        gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        norm_loss = weight_matrix * torch.mean((pred_flux - gt_flux) ** 2, dim=1)*train_mask
        norm_loss = norm_loss.sum(-1).mean()  ##TODO: Why do not just mean this

        # angle loss
        mask = train_mask * mask
        pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        # angle_loss = weight_matrix * (torch.acos(torch.sum(pred_flux * gt_flux, dim=1))) ** 2
        # angle_loss = angle_loss.sum(-1).mean()
        angle_loss = (1 - torch.cosine_similarity(pred_flux, gt_flux, dim=1))
        angle_loss = angle_loss[mask].mean()

        return norm_loss, angle_loss

    @staticmethod
    def get_poly_energy(energy_field, img_poly, ind, h, w):
        img_poly = img_poly.clone().float()
        img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
        img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

        batch_size = energy_field.size(0)
        gcn_feature = torch.zeros([img_poly.size(0), energy_field.size(1), img_poly.size(1)]).to(img_poly.device)
        for i in range(batch_size):
            poly = img_poly[ind == i].unsqueeze(0)
            gcn_feature[ind == i] = torch.nn.functional.grid_sample(energy_field[i:i + 1], poly)[0].permute(1, 0, 2)
        return gcn_feature

    def loss_energy_regularization(self, energy_field, img_poly, inds, h, w):
        energys = []
        for i, py in enumerate(img_poly):
            energy = self.get_poly_energy(energy_field.unsqueeze(1), py, inds, h, w)
            energys.append(energy.squeeze(1).sum(-1))

        regular_loss = torch.tensor(0.)
        energy_loss = torch.tensor(0.)
        for i, e in enumerate(energys[1:]):
            regular_loss += torch.clamp(e - energys[i], min=0.0).mean()
            ##TODO: Understand this energy loss
            energy_loss += torch.where(e <= 0.01, torch.tensor(0.), e).mean()

        return (energy_loss + regular_loss) / len(energys[1:])

    def forward(self, input_dict, output_dict, eps=None):
        """
        Calculate boundary proposal network loss
        """
        # tr_mask = tr_mask.permute(0, 3, 1, 2).contiguous()

        fy_preds = output_dict["fy_preds"]
        py_preds = output_dict["py_preds"]
        inds = output_dict["inds"]

        train_mask = input_dict["train_mask"]
        tr_mask = input_dict["tr_mask"] > 0
        distance_field = input_dict["distance_field"]
        direction_field = input_dict["direction_field"]
        ##TODO: Understanding `weight_matrix`
        weight_matrix = input_dict["weight_matrix"]
        gt_tags = input_dict["gt_points"]

        # # scake the prediction map
        # fy_preds = F.interpolate(fy_preds, scale_factor=cfg.scale, mode='bilinear')

        if cfg.scale > 1:
            ##TODO: warning that this squeeze will not work for batch size 1
            train_mask = F.interpolate(train_mask.float().unsqueeze(1),
                                       scale_factor=1/cfg.scale, mode='bilinear').squeeze(1).bool()
            tr_mask = F.interpolate(tr_mask.float().unsqueeze(1),
                                    scale_factor=1/cfg.scale, mode='bilinear').squeeze(1).bool()

            distance_field = F.interpolate(distance_field.unsqueeze(1),
                                           scale_factor=1/cfg.scale, mode='bilinear').squeeze(1)
            direction_field = F.interpolate(direction_field,
                                            scale_factor=1 / cfg.scale, mode='bilinear')
            weight_matrix = F.interpolate(weight_matrix.unsqueeze(1),
                                          scale_factor=1/cfg.scale, mode='bilinear').squeeze(1)
    
        # Pixel class loss
        # cls_loss = self.cls_ohem(fy_preds[:, 0, :, :], tr_mask.float(), train_mask)
        ##TODO: The code does not use OHEM
        cls_loss = self.BCE_loss(fy_preds[:, 0, :, :], tr_mask.float())
        cls_loss = torch.mul(cls_loss, train_mask.float()).mean()

        # distance field loss
        dis_loss = self.MSE_loss(fy_preds[:, 1, :, :], distance_field)
        dis_loss = torch.mul(dis_loss, train_mask.float())
        dis_loss = self.single_image_loss(dis_loss, distance_field)

        # direction field loss
        norm_loss, angle_loss = self.loss_calc_flux(fy_preds[:, 2:4, :, :], direction_field,
                                                    weight_matrix, tr_mask, train_mask)

        # boundary point loss
        point_loss = self.PolyMatchingLoss(py_preds[1:], gt_tags[inds])

        # Minimum energy loss regularization
        h, w = distance_field.size(1) * cfg.scale, distance_field.size(2) * cfg.scale
        energy_loss = self.loss_energy_regularization(distance_field, py_preds, inds[0], h, w)

        ##TODO: Modify and make these params dynamic
        if eps is None:
            alpha = cfg.alpha; beta = cfg.beta; theta = cfg.theta; gama = cfg.gama
        else:
            alpha = cfg.alpha; beta = cfg.beta; theta = cfg.theta
            gama = 0.1*torch.sigmoid(torch.tensor((eps - cfg.max_epoch) / cfg.max_epoch))
        loss = alpha * cls_loss + beta*dis_loss + theta*(norm_loss + angle_loss) + gama*(point_loss + energy_loss)
        
        loss_dict = {
            'cls_loss': alpha*cls_loss,
            'distance loss': beta*dis_loss,
            'dir_loss': theta*(norm_loss + angle_loss),
            'norm_loss': theta*norm_loss,
            'angle_loss': theta*angle_loss,
            'point_loss': gama*point_loss,
            'energy_loss': gama*energy_loss,
        }

        if cfg.enable_autofocus:
            # calculate loss for autofocus
            autofocus_preds = output_dict["autofocus_preds"]
            focus_mask = input_dict["flattened_focus_mask"]

            ##TODO: Loss NaN
            focus_loss = self.focus_loss(autofocus_preds, focus_mask)
            foc_weight = cfg.foc_weight
            loss_dict['focus_loss'] = foc_weight*focus_loss
            loss = loss + foc_weight * focus_loss

        loss_dict['total_loss'] = loss

        return loss_dict