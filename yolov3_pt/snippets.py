#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/24 21:33
@File:          snippets.py
'''

import math
import numpy as np

from nms import nms
from Config import Config

def yolo_head(feats, anchors_per_location, model_input_shape):
    feats = feats.transpose((0, 3, 4, 2, 1))
    dtype = feats.dtype
    num_anchors_per_location = len(anchors_per_location)
    anchors_per_location = np.reshape(anchors_per_location, [1, 1, 1, num_anchors_per_location, 2])

    grid_shape = np.shape(feats)[1:3]
    grid_x = np.tile(np.reshape(np.arange(grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid_y = np.tile(np.reshape(np.arange(grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    grid = grid.astype(dtype)[None, ...]

    box_xy = ((feats[..., :2] + grid) / grid_shape[::-1]).astype(dtype)
    box_wh = (np.exp(feats[..., 2:4]) * anchors_per_location / model_input_shape[::-1]).astype(dtype)
    box_confidence = feats[..., 4:5].astype(dtype)
    box_class_probs = feats[..., 5:].astype(dtype)

    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, model_input_shape, image_shape):
    dtype = box_xy.dtype

    min_scale = min([model_input_shape[i] / image_shape[i] for i in range(2)])
    new_shape = [math.floor(image_shape[i] * min_scale + 0.5) for i in range(2)]
    offset = [(model_input_shape[i] - new_shape[i]) / 2 / model_input_shape[i] for i in range(2)]
    scale = [model_input_shape[i] / new_shape[i] for i in range(2)]

    box_xy = (box_xy - offset[::-1]) * scale[::-1]
    box_wh *= scale[::-1]

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes[..., [0, 2]] *= image_shape[1]
    boxes[..., [1, 3]] *= image_shape[0]

    return boxes.astype(dtype)

def yolo_boxes_and_scores(feats, anchors_per_location, num_classes, model_input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors_per_location, model_input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, model_input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def get_detect_rsts(outputs,
                    anchors,
                    num_classes,
                    model_input_shape,
                    image_shape,
                    score_threshold=.6,
                    iou_threshold=.5):
    num_layers = len(outputs)

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(outputs[l], anchors[Config.ANCHOR_MASK[l]], num_classes,
                                                    model_input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)

    boxes = np.round(boxes).astype('int32')
    boxes[:, [0, 2]] = np.minimum(np.maximum(boxes[:, [0, 2]], 0), image_shape[1])
    boxes[:, [1, 3]] = np.minimum(np.maximum(boxes[:, [1, 3]], 0), image_shape[0])

    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]
    filter_index = np.logical_and(boxes_w > 1, boxes_h > 1)
    boxes = boxes[filter_index]
    box_scores = box_scores[filter_index]

    mask = (box_scores >= score_threshold)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = boxes[mask[:, c]]
        class_box_scores = box_scores[:, c][mask[:, c]]
        nms_index = nms(class_boxes, class_box_scores, iou_threshold=iou_threshold, max_num_boxes=Config.MAX_NUM_BOXES)
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        classes = np.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

def get_class_names(classes_path='model_data/voc_classes.txt'):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path='model_data/voc_anchors.txt'):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [[float(e) for e in wh.split(',')] for wh in anchors.split()]
    return np.array(anchors)
