#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/22 22:11
@File:          CollateFn.py
'''

import numpy as np
import torch

from Config import Config

def bbox_iou(true_box, anchors):
    min_xy = np.maximum(true_box[:, :2], anchors[:, :2])
    max_xy = np.minimum(true_box[:, 2:4], anchors[:, 2:4])
    inter_area = np.maximum(max_xy[:, 0] - min_xy[:, 0], 0) * np.maximum(max_xy[:, 1] - min_xy[:, 1], 0)
    union_area = ((true_box[:, 2] - true_box[:, 0]) * (true_box[:, 3] - true_box[:, 1]) +
                  (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]) - inter_area)
    return inter_area / union_area

class CollateFn:
    def __init__(self,
                 num_classes,
                 anchors,
                 model_input_shape=Config.MODEL_INPUT_SHAPE,
                 anchor_mask=Config.ANCHOR_MASK,
                 downsampling_scales=Config.DOWNSAMPLING_SCALES,
                 ignore_threshold=0.5):
        self.num_classes = num_classes
        self.model_input_shape = np.array(model_input_shape)
        self.anchors_per_location = [anchors[am] for am in anchor_mask]
        self.num_anchors_per_location = len(anchor_mask[0])
        self.num_layers = len(anchor_mask)
        self.feat_shapes = [np.ceil(self.model_input_shape / ds).astype('int32') for ds in downsampling_scales]
        self.downsampling_scales = downsampling_scales
        self.ignore_threshold = ignore_threshold

    def __call__(self, data):
        batch_size = len(data)

        batch_image = [t[0].transpose(2, 0, 1) for t in data]
        batch_image = torch.from_numpy(np.array(batch_image))

        batch_true_bboxes = [t[-1] for t in data]
        targets = [
            np.zeros(shape=(batch_size, h, w, self.num_anchors_per_location, self.num_classes + 5), dtype='float32')
            for h, w in self.feat_shapes]
        bbox_loss_scales = []
        bbox_loss_scales_mem = [
            np.zeros(shape=(batch_size, h, w, self.num_anchors_per_location, 2), dtype='float32')
            for h, w in self.feat_shapes]
        ignore_masks = [
            np.zeros(shape=(batch_size, h, w, self.num_anchors_per_location, 1), dtype='float32')
            for h, w in self.feat_shapes]

        for l in range(self.num_layers):
            target = targets[l]
            bbox_loss_scale_mem = bbox_loss_scales_mem[l]
            ignore_mask = ignore_masks[l]
            for b in range(batch_size):
                true_bboxes = batch_true_bboxes[b]
                for true_bbox in true_bboxes:
                    xy = (true_bbox[0:2] + true_bbox[2:4]) / 2
                    wh = true_bbox[2:4] - true_bbox[0:2]
                    xy = xy / self.downsampling_scales[l]
                    ij = xy.astype('int32')

                    bbox_shape = np.array([0, 0, wh[0], wh[1]], dtype='float32')[None, :]
                    anchor_shapes = np.concatenate([np.zeros((self.num_anchors_per_location, 2), dtype='float32'),
                                                    self.anchors_per_location[l].astype('float32')], axis=1)
                    iou = bbox_iou(bbox_shape, anchor_shapes)
                    best_n = np.argmax(iou)

                    target[b, ij[1], ij[0], best_n, 0:2] = xy - ij
                    target[b, ij[1], ij[0], best_n, 2:4] = np.log(wh / self.anchors_per_location[l][best_n])
                    target[b, ij[1], ij[0], best_n, 4] = 1
                    target[b, ij[1], ij[0], best_n, 5 + int(true_bbox[-1])] = 1

                    bbox_loss_scale_mem[b, ij[1], ij[0], best_n, :] = wh / self.model_input_shape[::-1]

                    ignore_mask[b, ij[1], ij[0], iou > self.ignore_threshold, 0] = 0

            bbox_loss_scales.append(2 - bbox_loss_scale_mem[..., 0:1] * bbox_loss_scale_mem[..., 1:2])

        targets = [torch.from_numpy(e) for e in targets]
        bbox_loss_scales = [torch.from_numpy(e) for e in bbox_loss_scales]
        ignore_masks = [torch.from_numpy(e) for e in ignore_masks]

        return batch_image, targets, bbox_loss_scales, ignore_masks
