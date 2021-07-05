#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/18 6:18
@File:          yolov3_model.py
'''

import torch
from torch import nn

class MakeLastLayer(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(MakeLastLayer, self).__init__()
        nonlinear = nn.LeakyReLU(negative_slope=0.2)
        self.com_stage = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nonlinear,
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nonlinear,
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nonlinear,
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nonlinear
        )
        self.x_stage = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nonlinear
        )
        self.y_stage = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nonlinear,
            nn.Conv2d(channels * 2, out_channels, 1, bias=False)
        )

    def forward(self, x):
        x = self.com_stage(x)
        x = self.x_stage(x)
        y = self.y_stage(x)
        return x, y

class YOLOBody(nn.Module):
    def __init__(self, BackBone, num_anchors_per_location, num_classes):
        super(YOLOBody, self).__init__()
        self.num_anchors_per_location = num_anchors_per_location
        out_channels = num_anchors_per_location * (num_classes + 5)
        nonlinear = nn.LeakyReLU(negative_slope=0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.backbone = BackBone()

        self.last_layer1 = MakeLastLayer(1024, 512, out_channels)

        self.conv_bn_act1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-12, momentum=1e-3),
            nonlinear
        )
        self.last_layer2 = MakeLastLayer(768, 256, out_channels)

        self.conv_bn_act2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-12, momentum=1e-3),
            nonlinear
        )
        self.last_layer3 = MakeLastLayer(384, 128, out_channels)

    def forward(self, x):
        fp1, fp2, fp3 = self.backbone(x)

        x, y1 = self.last_layer1(fp1)
        b, _, h, w = y1.size()
        y1 = y1.view(b, -1, self.num_anchors_per_location, h, w).permute(0, 3, 4, 2, 1).contiguous()
        y1 = self._act_pred(y1)

        x = self.conv_bn_act1(x)
        x = self.upsample(x)
        x = torch.cat([fp2, x], dim=1)

        x, y2 = self.last_layer2(x)
        b, _, h, w = y2.size()
        y2 = y2.view(b, -1, self.num_anchors_per_location, h, w).permute(0, 3, 4, 2, 1).contiguous()
        y2 = self._act_pred(y2)

        x = self.conv_bn_act2(x)
        x = self.upsample(x)
        x = torch.cat([fp3, x], dim=1)

        x, y3 = self.last_layer3(x)
        b, _, h, w = y3.size()
        y3 = y3.view(b, -1, self.num_anchors_per_location, h, w).permute(0, 3, 4, 2, 1).contiguous()
        y3 = self._act_pred(y3)

        return y1, y2, y3

    def _act_pred(self, pred):
        xy = torch.sigmoid(pred[..., 0:2])
        wh = pred[..., 2:4]
        conf = torch.sigmoid(pred[..., 4:5])
        cls = torch.sigmoid(pred[..., 5:])

        return torch.cat([xy, wh, conf, cls], dim=-1)
