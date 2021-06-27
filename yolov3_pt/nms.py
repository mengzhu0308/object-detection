#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/5/4 10:39
@File:          nms.py
'''

import numpy as np

def nms(bboxes, scores, iou_threshold=0.5, max_num_boxes=20):
    y1 = bboxes[:, 0]
    x1 = bboxes[:, 1]
    y2 = bboxes[:, 2]
    x2 = bboxes[:, 3]

    areas = (y2 - y1) * (x2 - x1)
    order = np.argsort(scores)[::-1]
    nms_index = []

    while np.size(order) > 0:
        i = order[0]
        nms_index.append(i)

        y_min = np.maximum(y1[i], y1[order[1:]])
        x_min = np.maximum(x1[i], x1[order[1:]])
        y_max = np.minimum(y2[i], y2[order[1:]])
        x_max = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0, y_max - y_min) * np.maximum(0, x_max - x_min)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        idx = np.nonzero(iou <= iou_threshold)[0]
        order = order[idx + 1]

    return nms_index[:max_num_boxes]
