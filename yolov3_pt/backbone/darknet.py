#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/6/27 5:52
@File:          darknet.py
'''

from torch import nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, num_blocks):
        super(ResBlock, self).__init__()
        self.num_blocks = num_blocks
        self.downsampling = ConvBNAct(in_channels, channels, 3, stride=2)
        for i in range(num_blocks):
            setattr(self, f'stage{i}',
                    nn.Sequential(ConvBNAct(channels, channels // 2, 1), ConvBNAct(channels // 2, channels, 3)))

    def forward(self, x):
        x = self.downsampling(x)
        identity = x
        for i in range(self.num_blocks):
            x = getattr(self, f'stage{i}')(x)
            x += identity
            identity = x

        return x

class DarkNetBody(nn.Module):
    def __init__(self):
        super(DarkNetBody, self).__init__()
        self.conv_bn_act = ConvBNAct(3, 32, 3)
        self.res_block1 = ResBlock(32, 64, 1)
        self.res_block2 = ResBlock(64, 128, 2)
        self.res_block3 = ResBlock(128, 256, 8)
        self.res_block4 = ResBlock(256, 512, 8)
        self.res_block5 = ResBlock(512, 1024, 4)

    def forward(self, x):
        x = self.conv_bn_act(x)
        x = self.res_block2(self.res_block1(x))
        fp3 = x = self.res_block3(x)
        fp2 = x = self.res_block4(x)
        fp1 = x = self.res_block5(x)

        return fp1, fp2, fp3