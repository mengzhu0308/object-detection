#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/20 8:24
@File:          YOLOLoss.py
'''

import torch
from torch import nn

from Config import Config

class YOLOLoss(nn.Module):
    def __init__(self, anchors, anchor_mask=Config.ANCHOR_MASK, model_input_shape=Config.MODEL_INPUT_SHAPE,
                 ignore_thresh=Config.IGNORE_THRESH):
        super(YOLOLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.anchor_mask = anchor_mask
        self.register_buffer('anchors', torch.as_tensor(anchors, dtype=torch.float32))
        self.register_buffer('model_input_shape', torch.as_tensor(model_input_shape, dtype=torch.int32))
        self.ignore_thresh = ignore_thresh

    def forward(self, raw_preds, targets):
        num_layers = len(raw_preds)
        batch_size = raw_preds[0].size(0)
        dtype, device = raw_preds[0].dtype, raw_preds[0].device
        model_input_shape_flip = torch.flip(self.model_input_shape.type(dtype=dtype), [0])
        losses = []

        for l in range(num_layers):
            object_mask = targets[l][..., 4:5]
            bbox_loss_scale = 2 - targets[l][..., 2:3] * targets[l][..., 3:4]
            grid_shape = raw_preds[l].size()[1:3]
            grid_shape_flip = torch.flip(torch.as_tensor(grid_shape, dtype=dtype, device=device), [0])
            grid = self._get_grid(grid_shape, device, dtype)
            raw_target_xy = targets[l][..., 0:2] * grid_shape_flip - grid[None, ...]
            raw_target_wh = targets[l][..., 2:4] * model_input_shape_flip / self.anchors[self.anchor_mask[l]]
            raw_target_wh = torch.where(raw_target_wh == 0, torch.zeros_like(raw_target_wh), torch.log(raw_target_wh))

            pred_xy = (raw_preds[l][..., 0:2] + grid[None, ...]) / grid_shape_flip
            pred_wh = torch.exp(raw_preds[l][..., 2:4]) * self.anchors[self.anchor_mask[l]] / model_input_shape_flip
            target_bbox = targets[l][..., 0:4]

            ignore_mask = []
            for b in range(batch_size):
                valid_target_bbox = target_bbox[b][object_mask[b][..., 0].type(dtype=torch.bool)]

                if valid_target_bbox.size(0) != 0:
                    iou = self._bbox_iou(valid_target_bbox[:, 0:2], valid_target_bbox[:, 2:4], pred_xy[b], pred_wh[b])
                    max_iou = torch.max(iou, dim=-1, keepdim=True)[0]
                    ignore_mask.append((max_iou < self.ignore_thresh).type(dtype=dtype)[None, ...])
                else:
                    ignore_mask.append(torch.ones_like(pred_xy[b][..., 0:1])[None, ...])
            ignore_mask = torch.cat(ignore_mask, dim=0)

            xy_loss = self.bce_loss(raw_preds[l][..., 0:2], raw_target_xy) * bbox_loss_scale * object_mask
            wh_loss = self.mse_loss(raw_preds[l][..., 2:4], raw_target_wh) * 0.5 * bbox_loss_scale * object_mask
            conf_loss = self.bce_loss(raw_preds[l][..., 4:5], object_mask) * (object_mask + (1 - object_mask) * ignore_mask)
            cls_loss = self.bce_loss(raw_preds[l][..., 5:], targets[l][..., 5:]) * object_mask

            xy_loss = torch.sum(xy_loss, dim=[1, 2, 3, 4])
            wh_loss = torch.sum(wh_loss, dim=[1, 2, 3, 4])
            conf_loss = torch.sum(conf_loss, dim=[1, 2, 3, 4])
            cls_loss = torch.sum(cls_loss, dim=[1, 2, 3, 4])

            loss = torch.mean(xy_loss + wh_loss + conf_loss + cls_loss)

            losses.append(loss)

        return sum(losses)

    def _get_grid(self, grid_shape, device, dtype):
        grid_y = torch.arange(0, grid_shape[0]).view(-1, 1, 1, 1).repeat(1, grid_shape[1], 1, 1)
        grid_x = torch.arange(0, grid_shape[1]).view(1, -1, 1, 1).repeat(grid_shape[0], 1, 1, 1)
        grid = torch.cat([grid_x, grid_y], dim=-1)

        return grid.to(device=device, dtype=dtype)

    def _bbox_iou(self, target_xy, target_wh, pred_xy, pred_wh):
        target_xy, target_wh = target_xy.view(1, 1, 1, -1, 2), target_wh.view(1, 1, 1, -1, 2)
        pred_xy, pred_wh = pred_xy[..., None, :], pred_wh[..., None, :]

        inter_mins = torch.maximum(pred_xy - pred_wh / 2, target_xy - target_wh / 2)
        inter_maxs = torch.minimum(pred_xy + pred_wh / 2, target_xy + target_wh / 2)
        inter_wh = torch.clamp_min(inter_maxs - inter_mins, 0)

        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        target_area = target_wh[..., 0] * target_wh[..., 1]

        return inter_area / (pred_area + target_area - inter_area)
