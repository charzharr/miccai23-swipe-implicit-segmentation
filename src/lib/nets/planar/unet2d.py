""" Module models/planar/unet2d.py (By: Charley Zhang, 2022)

Adapted from: https://github.com/milesial/Pytorch-UNet/tree/master/unet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basemodel import BaseModel

norm_layer = nn.BatchNorm2d  # used in UNet's DoubleConv


# ========================================================================== #
# * ### * ### * ### *          UNet Components           * ### * ### * ### * #
# ========================================================================== #


class DoubleConv(nn.Module):
    """ (Conv => [BN] => ReLU) * 2 """

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            norm_factory, 
            act_factory,
            mid_channels=None):
        
        self.norm = norm_factory
        self.act = act_factory
        
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, 
                      bias=False),
            self.norm.create(mid_channels),
            self.act.create(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, 
                      bias=False),
            self.norm.create(out_channels),
            self.act.create(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """ Downscaling with maxpool then double conv. """

    def __init__(
            self, 
            in_channels, 
            out_channels,
            norm_factory,
            act_factory):
        
        self.norm = norm_factory
        self.act = act_factory
        
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, self.norm, self.act)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """ Upscaling then double conv. """

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            norm_factory,
            act_factory,
            bilinear=True):
        
        self.norm = norm_factory
        self.act = act_factory
        
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, 
                                  mode='bilinear', 
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, self.norm, 
                                   self.act, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, 
                                   self.norm, self.act)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ========================================================================== #
# * ### * ### * ### *          UNet Definition           * ### * ### * ### * #
# ========================================================================== #


class UNet2d(BaseModel):
    
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            norm_factory, 
            act_factory,
            bilinear=True, 
            base=64,
            dropout_logits=0
            ):
        
        self.set_setting('norm', norm_factory)
        self.set_setting('act', act_factory)
        
        self.set_setting('in_channels', in_channels)
        self.set_setting('out_channels', out_channels)
        self.set_setting('bilinear', bilinear)
        self.set_setting('base', base)
        self.set_setting('dropout_logits', dropout_logits)
        
        # self.norm = norm_factory 
        # self.act = act_factory
        
        # self.in_channels = in_channels 
        # self.out_channels = out_channels
        # self.bilinear = bilinear
        # self.base = base
        # self.dropout_logits = dropout_logits
        
        super().__init__()
        
        self.inc = DoubleConv(in_channels, base, self.norm, self.act)
        self.down1 = Down(base, base * 2, self.norm, self.act)
        self.down2 = Down(base * 2, base * 4, self.norm, self.act)
        self.down3 = Down(base * 4, base * 8, self.norm, self.act)
        factor = 2 if bilinear else 1
        
        self.down4 = Down(base * 8, (base * 16) // factor, self.norm, self.act)
        
        self.up1 = Up(base * 16, (base * 8) // factor, 
                      self.norm, self.act, bilinear=bilinear)
        self.up2 = Up(base * 8, (base * 4) // factor, 
                      self.norm, self.act, bilinear=bilinear)
        self.up3 = Up(base * 4, (base * 2) // factor, 
                      self.norm, self.act, bilinear=bilinear)
        self.up4 = Up(base * 2, base, self.norm, self.act, bilinear=bilinear)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1) 
        
        self.print_settings()
    
    def forward(self, x, enc_only=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        if enc_only:
            return {'out': x5}
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return {
            'out': logits,
        }