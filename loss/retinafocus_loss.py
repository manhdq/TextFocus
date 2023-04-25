from .multibox_loss import MultiBoxLoss
from .focal_loss import FocalLoss


class RetinaFocusLoss(MultiBoxLoss):
    '''
    A simple combination of losses for RetinaFace and AutoFocus models.
    '''

    def __init__(self,
                 num_classes,
                 neg_pos,
                 variance,
                 cfg):
        super().__init__(num_classes=num_classes,
                         neg_pos=neg_pos,
                         variance=variance,
                         cfg=cfg)

        self.focus_criterion = FocalLoss(gamma=cfg['focal_gamma'], ignore_index=-1)

    def forward(self,
                retina_preds,
                retina_priors,
                retina_trgs,
                focus_preds,
                focus_trgs):
        loss_l, loss_r, loss_c, loss_landm \
            = super().forward(retina_preds, retina_priors, retina_trgs)
        loss_f = self.focus_criterion(focus_preds, focus_trgs)

        return loss_l, loss_r, loss_c, loss_landm, loss_f