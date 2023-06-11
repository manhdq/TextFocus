# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyMatchingLoss(nn.Module):
    def __init__(self, pnum, device, loss_type="L1"):
        super(PolyMatchingLoss, self).__init__()

        self.pnum = pnum
        self.device = device
        self.loss_type = loss_type
        self.smooth_L1 = F.smooth_l1_loss
        self.L2_loss = torch.nn.MSELoss(reduce=False, size_average=False)

        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx
        
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)
        self.feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2).detach()
        print(self.feature_id.shape)

    def match_loss(self, pred, gt):
        batch_size = pred.shape[0]
        feature_id = self.feature_id.expand(batch_size, self.feature_id.size(1), 2)

        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, self.pnum, self.pnum, 2)
        pred_expand = pred.unsqueeze(1)

        ##TODO: UserWarning: Using a target size (torch.Size([6, 20, 20, 2])) that is different to the input size (torch.Size([6, 1, 20, 2])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
        if self.loss_type == "L2":
            dis = self.L2_loss(pred_expand, gt_expand)
            dis = dis.sum(3).sqrt().mean(2)
        elif self.loss_type == "L1":
            dis = self.smooth_L1(pred_expand, gt_expand, reduction='none')
            dis = dis.sum(3).mean(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)

        return min_dis
        
    def forward(self, pred_list, gt):
        loss = torch.tensor(0.)
        for pred in pred_list:
            loss += torch.mean(self.match_loss(pred, gt))

        return loss / torch.tensor(len(pred_list))