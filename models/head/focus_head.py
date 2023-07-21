import torch.nn as nn
import torch
import torch.nn.functional as F

from ..loss import build_loss, iou


class AutoFocus(nn.Module):

    def __init__(self, in_channels, focus_layer_choice, loss_focus):
        super().__init__()

        self.focus_layer_choice = focus_layer_choice

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=256,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        self.conv_1_relu = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=(1, 1))
        self.conv_2_relu = nn.ReLU()

        self.conv_3 = nn.Conv2d(in_channels=256,
                                out_channels=2,
                                kernel_size=(1, 1))

        self.focus_loss = build_loss(loss_focus)

    def forward(self, data):
        out = self.conv_1_relu(self.conv_1(data))
        out = self.conv_2_relu(self.conv_2(out))
        out = self.conv_3(out)

        return out

    def loss(self, focus_preds, focus_masks, flattened_focus_masks):
        flattened_focus_preds = torch.reshape(focus_preds,
                                    shape=(focus_preds.shape[0], 2, -1))
        focus_loss = self.focus_loss(flattened_focus_preds, flattened_focus_masks)

        focus_preds = F.softmax(focus_preds, dim=1)[:, 1]
        training_focus_masks = (focus_masks != -1).long()
        focus_masks = (focus_masks == 1).long()
        iou_focus = iou(
            (focus_preds > 0.5).long(), focus_masks,
            training_focus_masks, reduce=False
        )
        
        losses = dict(loss_focus=focus_loss, iou_focus=iou_focus)

        return losses