#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/2/1 14:14
@File:          DataGenerator.py
'''

import numpy as np

from sampler import BatchSampler

from snippets import NUM_LAYERS, DOWNSAMPLING_SCALE, ANCHOR_MASK, MODEL_INPUT_SHAPE
from snippets import get_anchors, get_class_names

def preprocess_true_boxes(true_boxes, model_input_shape, anchors, num_classes):
    num_anchors = len(anchors) // NUM_LAYERS

    true_boxes = np.array(true_boxes, dtype='float32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy / model_input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / model_input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [model_input_shape // DOWNSAMPLING_SCALE[l] for l in range(NUM_LAYERS)]
    targets = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], num_anchors, 5 + num_classes),
                        dtype='float32') for l in range(NUM_LAYERS)]

    anchors = anchors[None, ...]
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue

        wh = wh[..., None, :]
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(NUM_LAYERS):
                if n in ANCHOR_MASK[l]:
                    j = np.floor(true_boxes[b, t, 0] * grid_shapes[l][0]).astype('int64')
                    i = np.floor(true_boxes[b, t, 1] * grid_shapes[l][1]).astype('int64')
                    k = ANCHOR_MASK[l].index(n)
                    c = true_boxes[b, t, 4].astype('int64')
                    targets[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    targets[l][b, j, i, k, 4] = 1
                    targets[l][b, j, i, k, 5 + c] = 1

    return targets

class BaseDataGenerator:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        super(BaseDataGenerator, self).__init__()
        self.dataset = dataset
        self.index_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self._sampler_iter = iter(self.index_sampler)

    @property
    def sampler_iter(self):
        return self._sampler_iter

    def __len__(self):
        return len(self.index_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        return self._next_data()

    def _next_index(self):
        try:
            index = next(self._sampler_iter)
        except StopIteration:
            self._sampler_iter = iter(self.index_sampler)
            index = next(self._sampler_iter)

        return index

    def _next_data(self):
        raise NotImplementedError

class DataGenerator(BaseDataGenerator):
    def __init__(self, dataset, **kwargs):
        super(DataGenerator, self).__init__(dataset, **kwargs)

    def _next_data(self):
        index = self._next_index()
        batch_images, batch_boxes = [], []
        for idx in index:
            image, boxes = self.dataset[idx]
            batch_images.append(image)
            batch_boxes.append(boxes)

        batch_images, batch_boxes = np.array(batch_images), np.array(batch_boxes)
        
        true_boxes = preprocess_true_boxes(batch_boxes, MODEL_INPUT_SHAPE, get_anchors(), len(get_class_names()))

        return [*true_boxes, batch_images], None
