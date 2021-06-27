#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/5/17 7:55
@File:          Config.py
'''

class Config:
    MODEL_INPUT_SHAPE = (416, 480)  # 适合VOC数据集
    NUM_LAYERS = 3
    DOWNSAMPLING_SCALE = [32, 16, 8]
    NUM_CLUSTER = 9
    ANCHOR_MASK = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    MAX_NUM_BOXES = 8