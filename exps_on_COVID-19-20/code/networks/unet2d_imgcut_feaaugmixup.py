# -*- coding: utf-8 -*-
"""
An implementation of the U-Net paper:
    Olaf Ronneberger, Philipp Fischer, Thomas Brox:
    U-Net: Convolutional Networks for Biomedical Image Segmentation. 
    MICCAI (3) 2015: 234-241
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
"""


import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

#!!!
import numpy as np
#!!!

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet2D(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet2D, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, mode='test',x2=None,alpha=0.4):
        if mode == 'test':
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            #x = self.relu(x)
            return x
        else:            
            x1_1 = self.inc(x)
            x1_2 = self.down1(x1_1)
            x1_3 = self.down2(x1_2)
            x1_4 = self.down3(x1_3)
            x1_5 = self.down4(x1_4)
            
            x1_out = self.up1(x1_5, x1_4)
            x1_out = self.up2(x1_out, x1_3)
            x1_out = self.up3(x1_out, x1_2)
            x1_out = self.up4(x1_out, x1_1)
            x1_out = self.outc(x1_out)
            
            #!!!feature-level cross-set mixup
            if x2 is None:
                bs1 = x1_5.size(0) // 2
                x15 = x1_5[:bs1]
                x14 = x1_4[:bs1]
                x13 = x1_3[:bs1]
                x12 = x1_2[:bs1]
                x11 = x1_1[:bs1]
                newindex  = torch.randperm(x1_5.size(0)).cuda()
                x2_5 = x1_5[newindex]
                lambda1 = torch.Tensor([np.random.beta(alpha, alpha)]).cuda()
                lambda2 = torch.Tensor([1.0-lambda1]).cuda()
                mixup_x5 = x15.mul(lambda1)+ x2_5.mul(lambda2)
                x1_out_mix = self.up1(mixup_x5, x14)
                x1_out_mix = self.up2(x1_out_mix, x13)
                x1_out_mix = self.up3(x1_out_mix, x12)
                x1_out_mix = self.up4(x1_out_mix, x11)
                x1_out_mix = self.outc(x1_out_mix)
                return x1_out, x1_out_mix, lambda1, lambda2, newindex
            else: 
                x2_1 = self.inc(x2)
                x2_2 = self.down1(x2_1)
                x2_3 = self.down2(x2_2)
                x2_4 = self.down3(x2_3)
                x2_5 = self.down4(x2_4)
                
                bs2 = x2_5.size(0)
                x15 = x1_5[:bs2]
                x14 = x1_4[:bs2]
                x13 = x1_3[:bs2]
                x12 = x1_2[:bs2]
                x11 = x1_1[:bs2]                
                
                lambda1 = torch.Tensor([np.random.beta(alpha, alpha)]).cuda()
                lambda2 = torch.Tensor([1.0-lambda1]).cuda()
                mixup_x5 = x15.mul(lambda1)+ x2_5.mul(lambda2)
                x1_out_mix = self.up1(mixup_x5, x14)
                x1_out_mix = self.up2(x1_out_mix, x13)
                x1_out_mix = self.up3(x1_out_mix, x12)
                x1_out_mix = self.up4(x1_out_mix, x11)
                x1_out_mix = self.outc(x1_out_mix)
                
                x2_out_mix = self.up1(mixup_x5, x2_4)
                x2_out_mix = self.up2(x2_out_mix, x2_3)
                x2_out_mix = self.up3(x2_out_mix, x2_2)
                x2_out_mix = self.up4(x2_out_mix, x2_1)
                x2_out_mix = self.outc(x2_out_mix)
                
                return x1_out, x1_out_mix, x2_out_mix, lambda1, lambda2
