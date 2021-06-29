#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/23 6:37
@File:          train.py
'''

import os
from collections import Counter
import math
import numpy as np
import cv2
from copy import deepcopy
import torch
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader

from Config import Config
from snippets import get_anchors, get_class_names, get_detect_rsts
from backbone.darknet import DarkNetBody
from yolov3_model import YOLOBody
from YOLOLoss import YOLOLoss

from ODDataset import ODDataset
from CollateFn import CollateFn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_batch_size = 32
    init_lr = 0.001
    epochs = 150
    initial_epoch = 0
    anchors = get_anchors()
    num_anchors = len(anchors)
    num_anchors_per_location = num_anchors // Config.NUM_LAYERS
    class_names = get_class_names()
    num_classes = len(class_names)
    space = '  '

    train_dataset = ODDataset('model_data/voc2012_train.txt')
    collate_fn = CollateFn(num_classes, anchors)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)

    model = YOLOBody(DarkNetBody, num_anchors_per_location, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=init_lr)
    criterion = YOLOLoss().to(device)

    num_train_batches = len(train_dataloader)
    num_train_examples = num_train_batches * train_batch_size

    def train():
        total_loss = 0.

        model.train()
        for i_batch, (images, targets, bbox_loss_scales, ignore_masks) in enumerate(train_dataloader):
            images = images.to(device)
            targets = [e.to(device) for e in targets]
            bbox_loss_scales = [e.to(device) for e in bbox_loss_scales]
            ignore_masks = [e.to(device) for e in ignore_masks]
            preds = model(images)
            loss = criterion(preds, targets, bbox_loss_scales, ignore_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            if math.isnan(loss_item) or math.isinf(loss_item):
                print('nan or inf')
                break

            print(f'\r{i_batch + 1}/{num_train_batches}{space}loss: {loss_item:.5f}',
                  end='\n' if i_batch + 1 >= num_train_batches else '',
                  flush=True)

            total_loss += loss.item()

        return total_loss / num_train_batches


    def resize(image):
        ih, iw, c = image.shape
        h, w = Config.MODEL_INPUT_SHAPE

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

    def show_det_image():
        img = cv2.imread('images/src.jpg')
        src_img = deepcopy(img)
        image_shape = img.shape[:2]
        img = resize(img).astype('float32') / 255

        model.eval()
        with torch.no_grad():
            img = torch.from_numpy(img.transpose(2, 0, 1)[None, ...])
            img = img.to(device)
            preds = model(img)
            preds = [e.cpu().numpy() for e in preds]

        out_boxes, out_scores, out_classes = get_detect_rsts(preds, anchors, num_classes, Config.MODEL_INPUT_SHAPE, image_shape)

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
            font_h = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0][1]
            cv2.putText(src_img, label, (left, top + cnt * font_h), FONT, FONT_SCALE, (255, 0, 0),
                        thickness=FONT_THICKNESS)
            cv2.imwrite('images/dst.jpg', src_img)

    print(f'Train on {len(train_dataset)} samples with train_batch_size equalling {train_batch_size}. ')
    print(f'init_lr eualling {init_lr}')
    
    best_loss = math.inf
    last_epoch = initial_epoch + epochs
    for i_epoch in range(initial_epoch, last_epoch):
        print(f'Epoch {i_epoch + 1}/{last_epoch}')

        loss = train()
        
        if loss < best_loss:
            best_loss = loss
            show_det_image()

        if math.isnan(loss) or math.isinf(loss):
            print('nan or inf')
            break

        print(f'loss: {loss:.5f}')
