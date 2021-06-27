#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/24 9:27
@File:          iou.py
'''

import numpy as np

def iou(true_boxes, anchors):
    min_xy = np.maximum(true_boxes[..., :2], anchors[..., :2])
    max_xy = np.minimum(true_boxes[..., 2:4], anchors[..., 2:4])
    inter_area = np.maximum(max_xy[..., 0] - min_xy[..., 0], 0) * np.maximum(max_xy[..., 1] - min_xy[..., 1], 0)
    union_area = ((true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1]) +
                  (anchors[..., 2] - anchors[..., 0]) * (anchors[..., 3] - anchors[..., 1]) - inter_area)
    return inter_area / union_area