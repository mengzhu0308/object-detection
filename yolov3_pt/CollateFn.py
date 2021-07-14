#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/22 22:11
@File:          CollateFn.py
'''

import numpy as np
import torch

from Config import Config

class CollateFn:
    def __init__(self,
                 num_classes,
                 anchors,
                 model_input_shape=Config.MODEL_INPUT_SHAPE,
                 anchor_mask=Config.ANCHOR_MASK,
                 downsampling_scales=Config.DOWNSAMPLING_SCALES):
        self.num_classes = num_classes
        self.model_input_shape = np.array(model_input_shape)
        self.anchors = anchors
        self.anchor_mask = anchor_mask
        self.num_anchors_per_location = len(anchor_mask[0])
        self.num_layers = len(anchor_mask)
        self.feat_shapes = [np.ceil(self.model_input_shape / ds).astype('int32') for ds in downsampling_scales]

    def __call__(self, data):
        batch_size = len(data)

        batch_image = [t[0].transpose(2, 0, 1) for t in data]
        batch_image = torch.from_numpy(np.array(batch_image))
        batch_true_bboxes = [t[-1] for t in data]

        targets = [
            np.zeros(shape=(batch_size, h, w, self.num_anchors_per_location, self.num_classes + 5), dtype='float32')
            for h, w in self.feat_shapes]

        anchors = self.anchors[None, ...]
        anchor_maxes = anchors / 2
        anchor_mins = -anchor_maxes

        for b, true_bboxes in enumerate(batch_true_bboxes):
            if len(true_bboxes) == 0:
                continue

            bboxes_xy = (true_bboxes[..., 0:2] + true_bboxes[..., 2:4]) / 2
            bboxes_wh = true_bboxes[..., 2:4] - true_bboxes[..., 0:2]
            true_bboxes[..., 0:2] = bboxes_xy / self.model_input_shape[::-1]
            true_bboxes[..., 2:4] = bboxes_wh / self.model_input_shape[::-1]

            wh = bboxes_wh[:, None, :]
            bbox_maxes = wh / 2
            bbox_mins = -bbox_maxes

            intersect_mins = np.maximum(bbox_mins, anchor_mins)
            intersect_maxes = np.minimum(bbox_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            ious = intersect_area / (box_area + anchor_area - intersect_area)
            
            gt_argmax_ious = np.argmax(ious, axis=-1)
            for t, n in enumerate(gt_argmax_ious):
                for l in range(self.num_layers):
                    if n in self.anchor_mask[l]:
                        i = np.floor(true_bboxes[t, 0] * self.feat_shapes[l][1]).astype('int32')
                        j = np.floor(true_bboxes[t, 1] * self.feat_shapes[l][0]).astype('int32')
                        k = self.anchor_mask[l].index(n)
                        c = true_bboxes[t, 4].astype('int32')
                        targets[l][b, j, i, k, 0:4] = true_bboxes[t, 0:4]
                        targets[l][b, j, i, k, 4] = 1
                        targets[l][b, j, i, k, 5 + c] = 1

        targets = [torch.from_numpy(e) for e in targets]

        return batch_image, targets
