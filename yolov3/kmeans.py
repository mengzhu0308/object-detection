#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/5/3 7:57
@File:          kmeans.py
'''

import numpy as np

from snippets import MODEL_INPUT_SHAPE, NUM_CLUSTER

def resize(image_shape, bboxes):
    ih, iw = image_shape
    h, w = MODEL_INPUT_SHAPE

    if w * ih < h * iw:
        factor = w / iw
    else:
        factor = h / ih

    bboxes_w = bboxes[:, [2]] - bboxes[:, [0]]
    bboxes_h = bboxes[:, [3]] - bboxes[:, [1]]

    return np.round(np.concatenate([bboxes_w, bboxes_h], axis=-1) * factor).astype('int32')

def load_dataset(file='model_data/voc2012_train.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_bboxes = []
    for line in lines:
        s = line.strip().split()
        img_size = s[1].split(',')
        bboxes = s[2:]

        image_shape = [int(i) for i in img_size][::-1]
        bboxes = [bbox.split(',')[:-1] for bbox in bboxes]
        bboxes = [[int(i) for i in box] for box in bboxes]
        bboxes = resize(image_shape, np.array(bboxes))
        all_bboxes.append(bboxes)

    return np.concatenate(all_bboxes, axis=0)

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    inter_w = np.minimum(clusters[:, 0], box[0])
    inter_h = np.minimum(clusters[:, 1], box[1])

    if np.count_nonzero(inter_w == 0) > 0 or np.count_nonzero(inter_h == 0) > 0:
        raise ValueError("Box has no area")

    intersection = inter_w * inter_h
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def avg_iou(bboxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param bboxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(bbox, clusters)) for bbox in bboxes])

def kmeans(bboxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param bboxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = bboxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = bboxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(bboxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(bboxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def result2txt(rst, file='model_data/voc_anchors.txt'):
    line = [f'{e[0]},{e[1]}' for e in rst]
    line = ' '.join(line)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(line)

all_bboxes = load_dataset()
rst = kmeans(all_bboxes, NUM_CLUSTER)
rst = rst[np.lexsort(rst.T[0, None])][::-1]

print("Accuracy: {:.2f}%".format(avg_iou(all_bboxes, rst) * 100))
print("Boxes:\n {}".format(rst))

result2txt(rst)