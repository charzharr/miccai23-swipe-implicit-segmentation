""" 
3D version of Res2Net (v1b) encoder with a UNet-like decoder.
 - 3D Res2UNet-50 (26w x 4s): 39M params (39.2M w. DeepSup)
 - 3D Res2UNet-101 (26w x 4s): 67M params (M w. DeepSup)

Adapted from their official git repo:
https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net_v1b.py
"""

import pathlib
import math
from os import stat
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

if __name__ == '__main__':
    import sys 
    curr_path = pathlib.Path(__file__).parent.absolute()
    sys.path.append(str(curr_path.parent.parent.parent))

from lib.nets.basemodel import BaseModel
from lib.nets.component_factories import NormFactory3d, ActFactory


__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b']


EPS = 1e-5


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 baseWidth=26, scale=4, stype='normal', 
                 norm='batchnorm', act='relu'):
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

        self.norm = NormFactory3d(norm)
        self.act = ActFactory(act)

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv3d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = self.norm.create(width*scale, eps=EPS)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool3d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv3d(width, width, kernel_size=3, stride=stride, 
                                 padding=1, bias=False))
          bns.append(self.norm.create(width, eps=EPS))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(width*scale, planes * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = self.norm.create(planes * self.expansion, eps=EPS)

        self.relu = self.act.create(inplace=True)
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
    def __init__(self, side_in_dim, bot_in_dim, out_dim,
                 norm='batchnorm', act='relu'):

        self.norm = NormFactory3d(norm)
        self.act = ActFactory(act)
                
        super(UpBlock, self).__init__()
        self.side_conv = nn.Conv3d(side_in_dim, out_dim, 1, bias=False)
        self.side_norm = self.norm.create(out_dim, eps=EPS)
        self.side_act = self.act.create()

        self.bot_conv = nn.Conv3d(bot_in_dim, out_dim, 1, bias=False)
        self.bot_norm = self.norm.create(out_dim, eps=EPS)
        self.bot_act = self.act.create()
        
        self.final_conv = nn.Conv3d(out_dim, out_dim, 3, padding=1,
                                    bias=False)
        self.final_norm = self.norm.create(out_dim, eps=EPS)
        self.final_act = self.act.create()
        
    def forward(self, side_in, bot_in):
        """
        Args:
            side_in (tensor): side input
            up_in (tensor): activated input from below that needs upsampling
        Returns:
            out (tensor): activated features
        """
        # process input from below (upsample by 2x)
        bot = F.interpolate(bot_in, scale_factor=2, mode='trilinear',
                            align_corners=True)
        bot = self.bot_act(self.bot_norm(self.bot_conv(bot)))

        side = self.side_act(self.side_norm(self.side_conv(side_in)))

        agg = bot + side 
        out = self.final_act(self.final_norm(self.final_conv(agg)))
        return out


class DeepSupBlock(nn.Module):
    def __init__(self, num_upsamples, in_channels, out_channels, 
                 use_deconv=True):
        super().__init__()
        assert num_upsamples >= 1
        
        self.num_upsamples = num_upsamples
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deconv = use_deconv
        
        modules_dict = OrderedDict()
        base_channels = max(out_channels, 16)
        for i in range(num_upsamples):
            scale_out_dims = base_channels * 2 ** (self.num_upsamples - i - 1)
            if i == 0:
                modules_dict['in_conv1'] = nn.Conv3d(in_channels, 
                    scale_out_dims, kernel_size=3, padding=1, bias=False)
                modules_dict['in_bn1'] = nn.BatchNorm3d(scale_out_dims, eps=EPS)
                modules_dict['in_act1'] = nn.ReLU()
                in_channels = scale_out_dims
            
            if self.use_deconv:
                modules_dict[f'scale{i+1}_up'] = nn.ConvTranspose3d(in_channels, 
                    scale_out_dims, kernel_size=4, stride=2, padding=1, 
                    bias=False)
                modules_dict[f'scale{i+1}_bn'] = nn.BatchNorm3d(scale_out_dims,
                                                                eps=EPS)
                modules_dict[f'scale{i+1}_act'] = nn.ReLU()
            else:
                modules_dict[f'scale{i+1}_up'] = nn.Upsample(scale_factor=2,
                    mode='trilinear', align_corners=True)
                
            if i == num_upsamples - 1: 
                modules_dict['final_conv'] = nn.Conv3d(scale_out_dims, 
                    out_channels, kernel_size=1, bias=True) 
            in_channels = scale_out_dims
                
        self.decoder = nn.Sequential(modules_dict)
    
    def forward(self, x):
        return self.decoder(x)


class Res2UNet3d(BaseModel):

    def __init__(self, block, layers, baseWidth=26, scale=4, 
                 in_channels=1, num_classes=2, prelinear_dropout=0,
                 norm='batchnorm', act='relu', deep_sup=True,
                 reduce_conv1_dstride=False):
        """
        Args:
            reduce_conv1_dstride (bool): if True, then change the stride of the
                1st conv (d=depth) in stem to (1, 2, 2) instead of (2, 2, 2)
        """
        
        self.prelinear_dropout = prelinear_dropout
        self.deep_sup = deep_sup
        self.num_classes = num_classes
        self.norm = NormFactory3d(norm)
        self.act = ActFactory(act)
        self.reduce_conv1_dstride = reduce_conv1_dstride
        
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale 
        
        conv1_stride = (1, 2, 2) if self.reduce_conv1_dstride else (2, 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, conv1_stride, 1, bias=False),
            self.norm.create(32, eps=EPS),
            self.act.create(inplace=True),
            nn.Conv3d(32, 32, 3, 1, 1, bias=False),
            self.norm.create(32, eps=EPS),
            self.act.create(inplace=True),
            nn.Conv3d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = self.norm.create(64, eps=EPS)
        self.relu = self.act.create()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.up16 = UpBlock(1024, 2048, 256, norm=norm, act=act)
        self.up8 = UpBlock(512, 256, 128, norm=norm, act=act)
        self.up4 = UpBlock(256, 128, 64, norm=norm, act=act)
        self.up2 = UpBlock(64, 64, 64, norm=norm, act=act)

        self.up2_conv = nn.Conv3d(64, 32, 3, 1, 1, bias=False)
        self.up2_norm = self.norm.create(32, eps=EPS)
        self.up2_act = self.act.create()
        self.final_conv = nn.Conv3d(32, num_classes, 1, bias=True)

        if self.deep_sup:
            self.up2_deepsup = DeepSupBlock(1, 64, num_classes)
            self.up4_deepsup = DeepSupBlock(2, 64, num_classes)
            self.up8_deepsup = DeepSupBlock(3, 128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        tot_params, tot_tparams = self.param_counts
        print(f'💠 Res2UNet_3d model initiated with n_classes={num_classes}, \n'
              f'   layers={layers}, base-width={baseWidth}, scale={scale}, \n'
              f'   in_chans={in_channels}, deep_sup={self.deep_sup}, '
                f'norm={norm}, act={act}, \n'
              f'   conv1_stride={conv1_stride}, \n'
              f'   params={tot_params:,}, trainable_params={tot_tparams:,}.')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool3d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv3d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                self.norm.create(planes * block.expansion, eps=EPS),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, 
                            downsample=downsample, stype='stage', 
                            baseWidth=self.baseWidth, scale=self.scale, 
                            norm=self.norm, act=self.act))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, 
                                scale=self.scale, norm=self.norm, act=self.act))

        return nn.Sequential(*layers)

    def forward(self, x, enc_only=False):
        
        # Encoder path
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
        
        if enc_only:
            return {
                'enc_out': x32,
                '4x':  x4,
                '8x':  x8,
                '16x': x16,
                '32x': x32
            }
        
        # Decoder path
        out16 = self.up16(x16, x32)
        out8 = self.up8(x8, out16)
        out4 = self.up4(x4, out8)
        out2 = self.up2(x2, out4)  # D,H//2,W//2 when conv1-stride=(1,2,2)

        out = self.up2_act(self.up2_norm(self.up2_conv(out2)))
        
        scale_factor = (1, 2, 2) if self.reduce_conv1_dstride else 2
        out = F.interpolate(out, scale_factor=scale_factor, 
                            mode='trilinear',
                            align_corners=True)
        out = self.final_conv(out)
        
        if self.deep_sup:
            return {
                'out': out,
                'enc_out': x32, 
                '2x': self.up2_deepsup(out2),
                '4x': self.up4_deepsup(out4), 
                '8x': self.up8_deepsup(out8),
            }
        else:
            return {
                'out': out,
                'enc_out': x32, 
                '4x':  x4,
                '8x':  x8,
                '16x': x16,
                '32x': x32
            }

def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    baseWidth = kwargs.pop('baseWidth') if 'baseWidth' in kwargs else 26
    scale = kwargs.pop('scale') if 'scale' in kwargs else 26
    model = Res2UNet3d(Bottle2neck, [3, 4, 6, 3], 
                       baseWidth=baseWidth, 
                       scale=scale, 
                       **kwargs)
    return model

def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    baseWidth = kwargs.pop('baseWidth') if 'baseWidth' in kwargs else 26
    scale = kwargs.pop('scale') if 'scale' in kwargs else 26
    model = Res2UNet3d(Bottle2neck, [3, 4, 23, 3], 
                       baseWidth=baseWidth, 
                       scale=scale, 
                       **kwargs)
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

# def load_state_dict(model, model_key):
#     print(f' * Res2Net1b loading pretrained ImageNet weights.')
#     # print(model.load_state_dict(model_zoo.load_url(model_urls[model_key])))
    
#     # My code after downloading model params
#     state_dict = torch.load(model_params[model_key], map_location='cpu')
#     if model.num_classes != 1000:
#         del state_dict['fc.weight']
#         del state_dict['fc.bias']
#     print(model.load_state_dict(state_dict, strict=False))

def get_model(layers, num_classes, pretrained=True, prelinear_dropout=0):
    layers = int(layers)
    if layers == 50:
        model = res2net50_v1b(pretrained=pretrained, num_classes=num_classes,
                              prelinear_dropout=prelinear_dropout)
    elif layers == 101:
        model = res2net101_v1b(pretrained=pretrained, num_classes=num_classes,
                               prelinear_dropout=prelinear_dropout)
    else:
        raise ValueError(f'{layers} layers is not supported right now.')
    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images = torch.rand(1, 1, 64, 64, 64).to(device)
    model = res2net50_v1b().to(device)
    out = model(images)
    
    for k, v in out.items():
        print(k, v.shape, v.min(), v.max())