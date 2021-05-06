#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/5/4 10:18
@File:          train.py
'''

import os
import math
import cv2
from collections import Counter
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import Callback

from snippets import MODEL_INPUT_SHAPE, DOWNSAMPLING_SCALE, NUM_LAYERS
from snippets import get_anchors, get_class_names, get_detect_rsts
from yolov3_model import create_yolov3

from Dataset import Dataset
from DataGenerator import DataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    train_batch_size = 32
    init_lr = 0.1
    epochs = 150
    anchors = get_anchors()
    num_anchors = len(anchors)
    class_names = get_class_names()
    num_classes = len(class_names)

    model = create_yolov3(anchors, num_classes, MODEL_INPUT_SHAPE)
    optimizer = SGD(learning_rate=init_lr, momentum=0.9, nesterov=True)
    model.compile(optimizer)

    train_dataset = Dataset('model_data/voc2012_train.txt')
    val_dataset = Dataset('model_data/voc2012_val.txt')
    train_generator = DataGenerator(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    def resize(image):
        ih, iw, c = image.shape
        h, w = MODEL_INPUT_SHAPE

        if w * ih < h * iw:
            factor = w / iw
        else:
            factor = h / ih

        nw, nh = int(iw * factor), int(ih * factor)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, c), (128, 128, 128), dtype=image.dtype)
        dy = (h - nh) // 2
        dx = (w - nw) // 2
        new_image[dy:dy + nh, dx:dx + nw, ...] = image

        return new_image

    def show():
        src_img = cv2.imread('detect_img/src.jpg')
        image_shape = src_img.shape[:2]
        input_image = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        input_image = resize(input_image).astype('float32') / 255
        h, w = MODEL_INPUT_SHAPE
        input_image = input_image[None, ...]
        targets = [np.zeros((1, math.ceil(h / DOWNSAMPLING_SCALE[l]), math.ceil(w / DOWNSAMPLING_SCALE[l]),
                             num_anchors // NUM_LAYERS, num_classes + 5), dtype='float32') for l in range(NUM_LAYERS)]
        preds = model.predict_on_batch([*targets, input_image])
        out_boxes, out_scores, out_classes = get_detect_rsts(preds, anchors, num_classes, MODEL_INPUT_SHAPE, image_shape)

        counter = Counter()
        FONT = cv2.FONT_HERSHEY_PLAIN
        FONT_SCALE = 1
        FONT_THICKNESS = 1
        for i, c in enumerate(out_classes):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            left, top, right, bottom = box
            key = f'{left}_{top}_{right}_{bottom}'
            counter.update(key)
            cnt = counter[key]
            if cnt == 1:
                cv2.rectangle(src_img, (left, top), (right, bottom), (0, 0, 255), thickness=1)
            font_h = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[1]
            cv2.putText(src_img, label, (left, top + (cnt - 1) * font_h * 2), FONT, FONT_SCALE, (255, 0, 0), 
                        thickness=FONT_THICKNESS)
            cv2.imwrite('detect_img/dst.jpg', src_img)

    class Show(Callback):
        def __init__(self):
            super(Show, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            show()

    print('Training......')

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[Show()],
        shuffle=False
    )
