#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/5/2 9:45
@File:          snippets.py
'''

import numpy as np

from nms import nms

MODEL_INPUT_SHAPE = (416, 480) # 适合VOC数据集
NUM_LAYERS = 3
DOWNSAMPLING_SCALE = [32, 16, 8]
NUM_CLUSTER = 9
ANCHOR_MASK = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
MAX_NUM_BOXES = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def yolo_head(feats, anchors, model_input_shape):
    num_anchors = len(anchors)
    anchors = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

    grid_shape = np.shape(feats)[1:3]
    grid_x = np.tile(np.reshape(np.arange(grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid_y = np.tile(np.reshape(np.arange(grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    grid = grid.astype(np.dtype(feats))

    box_xy = (sigmoid(feats[..., :2]) + grid) / grid_shape[::-1].astype(np.dtype(feats))
    box_wh = np.exp(feats[..., 2:4]) * anchors / model_input_shape[::-1].astype(np.dtype(feats))
    box_confidence = sigmoid(feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])

    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, model_input_shape, image_shape):
    model_input_shape_wh = model_input_shape[::-1].astype(np.dtype(box_xy))
    image_shape_wh = image_shape[::-1].astype(np.dtype(box_xy))
    new_shape_wh = np.round(image_shape_wh * np.min(model_input_shape_wh / image_shape_wh))
    offset = (model_input_shape_wh - new_shape_wh) / 2. / model_input_shape_wh
    scale = model_input_shape_wh / new_shape_wh
    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes *= np.concatenate([image_shape_wh, image_shape_wh], axis=-1)

    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, model_input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, model_input_shape)
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
        _boxes, _box_scores = yolo_boxes_and_scores(outputs[l], anchors[ANCHOR_MASK[l]], num_classes,
                                                    model_input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)

    mask = (box_scores >= score_threshold)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        mask_index = np.nonzero(mask[:, c])[0]
        class_boxes = boxes[mask_index]
        class_box_scores = box_scores[:, c][mask_index]
        nms_index = nms(class_boxes, class_box_scores, iou_threshold=iou_threshold, max_num_boxes=MAX_NUM_BOXES)
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        classes = np.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatenate(classes_, axis=0)

    boxes_ = np.round(boxes_).astype('int32')
    boxes_[:, :2] = np.maximum(0, boxes_[:, :2])
    boxes_[:, 2:3] = np.minimum(image_shape[1], boxes_[:, 2:3])
    boxes_[:, 3:4] = np.minimum(image_shape[0], boxes_[:, 3:4])

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