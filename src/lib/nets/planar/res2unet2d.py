""" Module res2unet2d.py By: Charley Zhang (2022)

Res2net encoder backbone + a light decoder. 

Adapted from Res2Net (v1b) official git repo:
https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net_v1b.py
"""

import os
from os import stat
import pathlib
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b']


model_params_path = '/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Models/'
model_params = {
    'res2net50_v1b_26w_4s': str(model_params_path / 'res2net50_v1b_26w_4s.pth'),
    'res2net101_v1b_26w_4s': str(model_params_path / 'res2net101_v1b_26w_4s.pth'),
}
model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}




class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, 
                                 padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UpBlock(nn.Module):
    """
    Upsamples via linear interpolation.
    """
    def __init__(self, side_in_dim, bot_in_dim, out_dim):
                
        super(UpBlock, self).__init__()
        self.side_conv = nn.Conv2d(side_in_dim, out_dim, 1, bias=False)
        self.side_norm = nn.BatchNorm2d(out_dim)
        self.side_act = nn.ReLU(inplace=True)

        self.bot_conv = nn.Conv2d(bot_in_dim, out_dim, 1, bias=False)
        self.bot_norm = nn.BatchNorm2d(out_dim)
        self.bot_act = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1,
                                    bias=False)
        self.final_norm = nn.BatchNorm2d(out_dim)
        self.final_act = nn.ReLU(inplace=True)
        
    def forward(self, side_in, bot_in):
        """
        Args:
            side_in (tensor): side input
            up_in (tensor): activated input from below that needs upsampling
        Returns:
            out (tensor): activated features
        """
        # process input from below (upsample by 2x)
        bot = F.interpolate(bot_in, scale_factor=2, mode='bilinear',
                                      align_corners=True)
        bot = self.bot_act(self.bot_norm(self.bot_conv(bot)))

        side = self.side_act(self.side_norm(self.side_conv(side_in)))

        agg = bot + side 
        out = self.final_act(self.final_norm(self.final_conv(agg)))
        return out


class Res2UNet(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000,
                 in_channels=1, prelinear_dropout=0):
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.prelinear_dropout = prelinear_dropout
        
        self.inplanes = 64
        super().__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.up16 = UpBlock(1024, 2048, 256)
        self.up8 = UpBlock(512, 256, 128)
        self.up4 = UpBlock(256, 128, 64)
        self.up2 = UpBlock(64, 64, 64)

        self.up2_conv = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.up2_norm = nn.BatchNorm2d(32)
        self.up2_act = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(32, num_classes, 1, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        tot_params = sum([p.numel() for p in self.parameters()])
        print(f'💠 Res2Netv1b model initiated with n_classes={num_classes}, \n'
              f'   layers={layers}, base-width={baseWidth}, scale={scale}\n'
              f'   params={tot_params:,}.')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, 
                            downsample=downsample, stype='stage', 
                            baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, 
                                scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x4 = self.maxpool(x2)

        x4 = self.layer1(x4)  # 4x down, activated
        x8 = self.layer2(x4)  # 8x down, activated
        x16 = self.layer3(x8)  # 16x down, activated
        x32 = self.layer4(x16)  # 32x down, activated
        
        if self.prelinear_dropout > 0:
            x32 = F.dropout(x32, p=self.prelinear_dropout, 
                            training=self.training)

        # Decoder path
        out16 = self.up16(x16, x32)
        out8 = self.up8(x8, out16)
        out4 = self.up4(x4, out8)
        out2 = self.up2(x2, out4)
        
        out = self.up2_act(self.up2_norm(self.up2_conv(out2)))
        prefinal_out = F.interpolate(out, scale_factor=2, mode='bilinear',
                                     align_corners=True)
        out = self.final_conv(prefinal_out)
        
        return {'out': out,
                '1x':  prefinal_out,
                '2x':  out2,
                '4x':  out4,
                '8x':  out8,
                '16x': out16,
                '32x': x32}


def res2unet50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2UNet(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        load_state_dict(model, 'res2net50_v1b_26w_4s')
    return model

def res2unet101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-101_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2UNet(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        load_state_dict(model, 'res2net101_v1b_26w_4s')
    return model

# def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
#     """Constructs a Res2Net-50_v1b_26w_4s model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
#     if pretrained:
#         load_state_dict(model, 'res2net50_v1b_26w_4s')
#     return model

# def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
#     """Constructs a Res2Net-50_v1b_26w_4s model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
#     if pretrained:
#         load_state_dict(model, 'res2net101_v1b_26w_4s')
#     return model

# def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
#     """Constructs a Res2Net-50_v1b_26w_4s model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
#     if pretrained:
#         load_state_dict(model, 'res2net152_v1b_26w_4s')
        
#     return model

def load_state_dict(model, model_key):
    # My code after downloading model params
    if model_key in model_params:
        print(f' 💬 Attempting to load model from key "({model_key})."')
        if os.path.isfile(model_params[model_key]):
            print(f' 💬 File found! Loading from "({model_params[model_key]})."')
            state_dict = torch.load(model_params[model_key], map_location='cpu')
    else:
        print(f' 💬 Res2Net1b loading {model_key} params from url.')
        state_dict = model_zoo.load_url(model_urls[model_key])
    
    if model.in_channels != 3:
        del state_dict['conv1.0.weight']
    if model.num_classes != 1000:
        del state_dict['fc.weight']
        del state_dict['fc.bias']
    print(model.load_state_dict(state_dict, strict=False))


def get_model(layers, num_classes, in_channels=1, 
              pretrained=True, prelinear_dropout=0):
    layers = int(layers)
    if layers == 50:
        model = res2unet50_v1b(pretrained=pretrained, num_classes=num_classes,
                              prelinear_dropout=prelinear_dropout,
                              in_channels=in_channels)
    elif layers == 101:
        model = res2unet101_v1b(pretrained=pretrained, num_classes=num_classes,
                               prelinear_dropout=prelinear_dropout,
                               in_channels=in_channels)
    else:
        raise ValueError(f'{layers} layers is not supported right now.')
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2unet101_v1b(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())