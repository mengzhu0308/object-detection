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
                 downsampling_scale=Config.DOWNSAMPLING_SCALE,
                 ignore_threshold=0.5):
        self.num_classes = num_classes
        self.model_input_shape = np.array(model_input_shape)
        self.anchors_per_location = [anchors[am] for am in anchor_mask]
        self.num_anchors_per_location = len(anchor_mask[0])
        self.num_layers = len(anchor_mask)
        self.feat_shapes = [self.model_input_shape // ds for ds in downsampling_scale]
        self.ignore_threshold = ignore_threshold

    def __call__(self, data):
        batch_size = len(data)

        batch_image = [t[0].transpose(2, 0, 1) for t in data]
        batch_image = torch.from_numpy(np.array(batch_image))

        batch_true_bboxes = [t[-1] for t in data]
        targets = [
            np.zeros(shape=(batch_size, self.num_classes + 5, self.num_anchors_per_location, h, w), dtype='float32')
            for h, w in self.feat_shapes]
        bbox_loss_scales = []
        bbox_loss_scales_mem = [
            np.zeros(shape=(batch_size, 2, self.num_anchors_per_location, h, w), dtype='float32')
            for h, w in self.feat_shapes]
        ignore_masks = [
            np.zeros(shape=(batch_size, 1, self.num_anchors_per_location, h, w), dtype='float32')
            for h, w in self.feat_shapes]

        for l in range(self.num_layers):
            target = targets[l]
            bbox_loss_scale_mem = bbox_loss_scales_mem[l]
            ignore_mask = ignore_masks[l]
            for b in range(batch_size):
                true_bboxes = batch_true_bboxes[b]
                for t in range(len(true_bboxes)):
                    xy = (true_bboxes[t][0:2] + true_bboxes[t][2:4]) / 2
                    wh = true_bboxes[t][2:4] - true_bboxes[t][0:2]

                    xy = xy / self.model_input_shape[::-1] * self.feat_shapes[l][::-1]
                    ij = xy.astype('int32')

                    bbox_shape = np.array([0, 0, wh[0], wh[1]], dtype='float32')[None, :]
                    anchor_shapes = np.concatenate([np.zeros((self.num_anchors_per_location, 2), dtype='float32'),
                                                    self.anchors_per_location[l].astype('float32')], axis=1)
                    iou_cur = bbox_iou(bbox_shape, anchor_shapes)
                    best_n = np.argmax(iou_cur)

                    target[b, 0:2, best_n, ij[1], ij[0]] = xy - ij
                    target[b, 2:4, best_n, ij[1], ij[0]] = np.log(wh / self.anchors_per_location[l][best_n] + 1e-12)
                    target[b, 4, best_n, ij[1], ij[0]] = 1
                    target[b, int(true_bboxes[t][-1]), best_n, ij[1], ij[0]] = 1

                    bbox_loss_scale_mem[b, :, best_n, ij[1], ij[0]] = wh / self.model_input_shape[::-1]

                    ignore_mask[b, 0, iou_cur > self.ignore_threshold, ij[1], ij[0]] = 0

            bbox_loss_scales.append(2 - bbox_loss_scale_mem[:, 0:1, ...] * bbox_loss_scale_mem[:, 1:2, ...])

        targets = [torch.from_numpy(e) for e in targets]
        bbox_loss_scales = [torch.from_numpy(e) for e in bbox_loss_scales]
        ignore_masks = [torch.from_numpy(e) for e in ignore_masks]

        return batch_image, targets, bbox_loss_scales, ignore_masks
