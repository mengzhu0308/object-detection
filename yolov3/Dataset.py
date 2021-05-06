'''
@Author:        ZM
@Date and Time: 2019/10/8 6:28
@File:          Dataset.py
'''

from skimage import io
import numpy as np
import cv2

from snippets import MAX_NUM_BOXES, MODEL_INPUT_SHAPE

def resize(image, bboxes):
    ih, iw, c = image.shape
    h, w = MODEL_INPUT_SHAPE

    if w * ih < h * iw:
        factor = w / iw
    else:
        factor = h / ih

    nw, nh = int(iw * factor), int(ih * factor)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    bboxes[:, :4] = (bboxes.astype('float32')[:, :4] * factor)
    bboxes = bboxes.astype('int32')

    new_image = np.full((h, w, c), (128, 128, 128), dtype=image.dtype)

    dy = (h - nh) // 2
    dx = (w - nw) // 2

    new_image[dy:dy + nh, dx:dx + nw, ...] = image
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + dx
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + dy

    return new_image, bboxes

def boxes_padding(boxes, max_num_boxes=20):
    box_data = np.zeros((max_num_boxes, 5), dtype='int32')

    if len(boxes) > 0:
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]

        if len(boxes) > max_num_boxes:
            boxes = boxes[:max_num_boxes]

        np.random.shuffle(boxes)

        box_data[:len(boxes)] = boxes

    return box_data

class Dataset:
    def __init__(self, annotation_file):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        image_paths, all_bboxes = [], []
        for line in lines:
            line = line.strip().split()
            image_path = line[0]
            bboxes = [[int(e) for e in bbox.split(',')] for bbox in line[2:]]
            image_paths.append(image_path)
            all_bboxes.append(bboxes)

        self.image_paths = image_paths
        self.all_bboxes = all_bboxes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = io.imread(self.image_paths[item])
        bboxes = np.array(self.all_bboxes[item])

        image, bboxes = resize(image, bboxes)

        image = image.astype('float32') / 255
        bboxes = boxes_padding(bboxes, max_num_boxes=MAX_NUM_BOXES)

        return image, bboxes
