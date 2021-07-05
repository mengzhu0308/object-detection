#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/20 8:24
@File:          YOLOLoss.py
'''

import torch
from torch import nn

class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, bbox_loss_scales, ignore_masks):
        num_layers = len(preds)
        losses = []
        for l in range(num_layers):
            losses.append(self._cal_loss(preds[l], targets[l], bbox_loss_scales[l], ignore_masks[l]))

        return sum(losses)

    def _cal_loss(self, pred, target, bbox_loss_scale, ignore_mask):
        object_mask = target[..., 4:5]

        xy_loss = self.bce_loss(pred[..., 0:2], target[..., 0:2]) * bbox_loss_scale * object_mask
        wh_loss = self.mse_loss(pred[..., 2:4], target[..., 2:4]) * bbox_loss_scale * object_mask
        conf_loss = self.bce_loss(pred[..., 4:5], object_mask) * (object_mask + (1 - object_mask) * ignore_mask)
        cls_loss = self.bce_loss(pred[..., 5:], target[..., 5:]) * object_mask

        xy_loss = torch.sum(xy_loss, dim=[1, 2, 3, 4])
        wh_loss = torch.sum(wh_loss, dim=[1, 2, 3, 4])
        conf_loss = torch.sum(conf_loss, dim=[1, 2, 3, 4])
        cls_loss = torch.sum(cls_loss, dim=[1, 2, 3, 4])

        loss = torch.mean(xy_loss + wh_loss + conf_loss + cls_loss)

        return loss
