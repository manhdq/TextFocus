from os import device_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_utils import match, log_sum_exp


##TODO: Ignore training object too tiny or too large in specific chip
class MultiBoxLoss(nn.Module):
    ''' Adapted from SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    '''

    def __init__(self,
                 num_classes,
                 neg_pos,
                 variance,
                 cfg,
                 device='cuda'):
        super(MultiBoxLoss, self).__init__()

        self.num_classes = num_classes
        self.negpos_ratio = neg_pos
        self.variance = variance
        self.cfg = cfg
        self.device = device

    def forward(self, predictions, priors, targets):
        '''Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes.
                conf shape: torch.size(batch_size, num_priors, num_classes)
                loc shape: torch.size(batch_size, num_priors, 4)

            priors (tensor): Prior Bboxes for certain image size.
                priors shape: torch.size(num_priors, 4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        '''

        loc_data, class_data, conf_data, landm_data = predictions
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # Match priors (default boxes) and groundtruth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        valid_lm_mask = torch.BoolTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            if len(truths) > 30:  # A cool trick to prevent data leaking during the training process
                truths, defaults, labels, landms = truths.cpu(
                ), defaults.cpu(), labels.cpu(), landms.cpu()
                match(self.cfg,
                    truths,
                    defaults,
                    self.variance,
                    labels,
                    landms,
                    loc_t,
                    conf_t,
                    landm_t,
                    valid_lm_mask,
                    idx)
                if self.device == 'cuda':
                    truths, defaults, labels, landms = map(
                            self._to_cuda, [truths, defaults, labels, landms])
            else:
                match(self.cfg,
                      truths,
                      defaults,
                      self.variance,
                      labels,
                      landms,
                      loc_t,
                      conf_t,
                      landm_t,
                      valid_lm_mask,
                      idx)
        
        if self.device == 'cuda':
            loc_t, conf_t, landm_t, valid_lm_mask = map(self._to_cuda, [loc_t, conf_t, landm_t, valid_lm_mask])
        
        # Compute landm loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        ##TODO: Only train landmarks on MR label (class == 1)
        ##TODO: Set option landmarks training (this auto enable train for text)
        pos1 = conf_t == torch.tensor(self.cfg['fg_class_id']).to(self.device)
        pos1 = torch.logical_and(pos1, valid_lm_mask)
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        class_t = conf_t.clone()
        pos = conf_t != torch.tensor(self.cfg['bg_class_id']).to(self.device)
        conf_t[pos] = 1

        # Compute localization loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # Compute localization loss for all classes except background (class_id == 0)
        # Positive labels (not background)
        ##TODO: Calculate loss in normalized form ??
        pos = conf_t != torch.tensor(0).to(self.device)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        # We have 2 classes for classification loss (0: Background, 1: Object)
        batch_conf = conf_data.view(-1, 2)
        loss_c = log_sum_exp(batch_conf) - \
            batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard negative mining
        loss_c[pos.view(-1, 1)] = 0  # Filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        num_neg[num_neg < self.cfg['min_num_neg']] = self.cfg['min_num_neg']
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence loss including positive and negative examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # We have 2 classes for classification loss (0: Background, 1: Object)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, 2)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Class loss
        num_classes = self.num_classes
        pos_idx = pos.unsqueeze(2).expand_as(class_data)
        neg_idx = neg.unsqueeze(2).expand_as(class_data)
        class_p = class_data[(pos_idx+neg_idx).gt(0)].view(-1, num_classes)
        targets_weighted = class_t[(pos+neg).gt(0)]
        loss_r = F.cross_entropy(class_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        N_neg = max(num_neg.data.sum().float(), 1)
        loss_l /= N
        loss_r = loss_r / (N + N_neg) * (self.negpos_ratio + 1)
        loss_c = loss_c / (N + N_neg) * (self.negpos_ratio + 1)
        loss_landm /= N1
        return loss_l, loss_r, loss_c, loss_landm


    @staticmethod
    def _to_cuda(tensor):
        return tensor.cuda()