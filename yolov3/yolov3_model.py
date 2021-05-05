#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/5/1 11:34
@File:          yolov3_model.py
'''

import math
import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras import Model

from snippets import ANCHOR_MASK, DOWNSAMPLING_SCALE, NUM_LAYERS

def conv_bn_act(x, filters, kernel_size, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x

def resblock(x, filters, nblocks):
    x = conv_bn_act(x, filters, 3, strides=2)
    identity = x
    for _ in range(nblocks):
        x = conv_bn_act(x, filters // 2, 1)
        x = conv_bn_act(x, filters, 3)
        x = add([identity, x])
        identity = x
    return x

def darknet_body(x):
    x = conv_bn_act(x, 32, 3)
    blocks = [1, 2, 8, 8, 4]
    for i, nblocks in enumerate(blocks):
        x = resblock(x, 64 * 2 ** i, nblocks)
    return x

def make_last_layers(x, filters, out_filters):
    for _ in range(2):
        x = conv_bn_act(x, filters, 1)
        x = conv_bn_act(x, filters * 2, 3)
    x = conv_bn_act(x, filters, 1)
    y = conv_bn_act(x, filters * 2, 3)
    y = conv_bn_act(y, out_filters, 1)
    return x, y

def yolo_bdoy(inputs, num_anchors, num_classes):
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))
    h, w = K.int_shape(y1)[1:3]
    y1 = Reshape((h, w, num_anchors, num_classes + 5))(y1)

    x = conv_bn_act(x, 256, 1)
    x = UpSampling2D(interpolation='bilinear')(x)
    x = concatenate([darknet.get_layer(name='add_19').output, x])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))
    h, w = K.int_shape(y2)[1:3]
    y2 = Reshape((h, w, num_anchors, num_classes + 5))(y2)

    x = conv_bn_act(x, 128, 1)
    x = UpSampling2D(interpolation='bilinear')(x)
    x = concatenate([darknet.get_layer(name='add_11').output, x])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    h, w = K.int_shape(y3)[1:3]
    y3 = Reshape((h, w, num_anchors, num_classes + 5))(y3)

    return y1, y2, y3

def yolo_head(feats, anchors, model_input_shape):
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(model_input_shape[::-1], K.dtype(feats))

    return grid, box_xy, box_wh

def box_iou(b1, b2):
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b1_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b1_xy - b2_wh_half
    b2_maxes = b1_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

class YOLOLoss(Layer):
    def __init__(self, model_input_shape, anchors, num_classes, ignore_thresh=0.5, output_axis=(3, 4, 5), **kwargs):
        super(YOLOLoss, self).__init__(**kwargs)
        self.model_input_shape = model_input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.output_axis = output_axis

    def call(self, inputs, **kwargs):
        loss = self.compute_loss(inputs)
        self.add_loss(loss)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, (list, tuple)):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs):
        num_layers = len(inputs) // 2
        targets = inputs[:num_layers]
        outputs = inputs[num_layers:]

        model_input_shape = K.constant(self.model_input_shape, dtype=K.dtype(targets[0]))
        grid_shapes = [K.cast(K.shape(outputs[l])[1:3], K.dtype(targets[0])) for l in range(num_layers)]
        loss = 0
        m = K.shape(outputs[0])[0]

        for l in range(num_layers):
            object_mask = targets[l][..., 4:5]
            true_class_probs = targets[l][..., 5:]

            raw_pred = outputs[l]
            grid, pred_xy, pred_wh = yolo_head(outputs[l], self.anchors[ANCHOR_MASK[l]], model_input_shape)
            pred_box = K.concatenate([pred_xy, pred_wh])

            # Darknet raw box to calculate loss.
            raw_true_xy = targets[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = K.log(targets[l][..., 2:4] * model_input_shape[::-1] / self.anchors[ANCHOR_MASK[l]])
            raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
            box_loss_scale = 2 - targets[l][..., 2:3] * targets[l][..., 3:4]

            # Find ignore mask, iterate over each of batch.
            ignore_mask_b0 = tf.TensorArray(K.dtype(targets[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, 'bool')

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(targets[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = box_iou(pred_box[b], true_box)
                best_iou = K.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < self.ignore_thresh, K.dtype(true_box)))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < m, loop_body, [0, ignore_mask_b0])
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            xy_loss = (object_mask * box_loss_scale *
                       K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True))
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            confidence_loss = ((object_mask + (1 - object_mask) * ignore_mask) *
                               K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True))
            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

            xy_loss = K.sum(xy_loss, axis=[1, 2, 3, 4])
            wh_loss = K.sum(wh_loss, axis=[1, 2, 3, 4])
            confidence_loss = K.sum(confidence_loss, axis=[1, 2, 3, 4])
            class_loss = K.sum(class_loss, axis=[1, 2, 3, 4])
            loss += K.mean(xy_loss + wh_loss + confidence_loss + class_loss)

        return loss

    def compute_output_shape(self, input_shape):
        if self.output_axis is None:
            return input_shape
        elif isinstance(self.output_axis, (list, tuple)):
            return [input_shape[i] for i in self.output_axis]
        else:
            return input_shape[self.output_axis]

    def get_config(self):
        config = {
            'model_input_shape': self.model_input_shape,
            'anchors': self.anchors,
            'num_classes': self.num_classes,
            'ignore_thresh': self.ignore_thresh,
            'output_axis': self.output_axis
        }
        base_config = super(YOLOLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def create_yolov3(anchors, num_classes, model_input_shape):
    num_anchors = len(anchors) // NUM_LAYERS
    h, w = model_input_shape
    inputs = Input(shape=(h, w, 3), dtype='float32')
    outputs = yolo_bdoy(inputs, num_anchors, num_classes)
    targets = [Input(shape=(math.ceil(h / DOWNSAMPLING_SCALE[l]),
                            math.ceil(w / DOWNSAMPLING_SCALE[l]),
                            num_anchors, num_classes + 5), dtype='float32') for l in range(NUM_LAYERS)]
    outputs = YOLOLoss(model_input_shape, anchors, num_classes)([*targets, *outputs])
    return Model([*targets, inputs], outputs)