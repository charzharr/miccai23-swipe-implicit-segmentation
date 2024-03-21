"""
Adapted from Res2Net (v1b) official git repo:
https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net_v1b.py
"""


import pathlib
import math
import numbers
from os import stat
from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.linalg as LA

from lib.nets.basemodel import BaseModel
from lib.nets.component_factories import NormFactory2d, ActFactory
from lib.nets.modules.droppath import DropPath


DOWN_STRIDES = (2, 2)
BLOCK_EXPANSION = 4                # orig: 4
STAGE_DIMS = [64, 128, 256, 512]   # orig: [64, 128, 256, 512], 1st must =64
                                   #  [64, 96, 128, 160]



# ========================================================================== #
# * ### * ### * ### *            Layer Modules           * ### * ### * ### * #
# ========================================================================== #


class Conv2d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            norm_factory,  # for compatibility
            act_factory,   # for compatibility
            kernel_size=3, 
            stride=1, 
            padding=1,
            groups=1, 
            bias=False
            ):
        super().__init__()
        
        self.stride = stride 
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        
    def forward(self, x):
        x = self.conv(x)
        return x


class ForwardHook():
    features = None
    
    def __init__(self, module): 
        self.hook = module.register_forward_hook(self.hook_fn)
        print(f'🪝 Registered in output of {module}')

    def hook_fn(self, module, input, output): 
        self.features = output

    def remove(self): 
        self.hook.remove()



# ========================================================================== #
# * ### * ### * ### *            Block Modules           * ### * ### * ### * #
# ========================================================================== #


class Bottleneck(nn.Module):
    """ Regular Bottleneck block with stochastic depth. """
    expansion = BLOCK_EXPANSION

    def __init__(
            self, 
            in_planes,   
            planes,  
            norm_factory,
            act_factory,
            stride=1, 
            downsample=None,
            groups=1,
            conv_layer=None, 
            drop_path_rate=0.,
            stochastic_depth=0.
            ):        
        
        super().__init__()
        
        self.stochastic_depth_rate = stochastic_depth
        self.drop_path_rate = drop_path_rate
        self.norm = norm_factory 
        self.act = act_factory
        
        conv_layer = conv_layer or Conv2d
        mid_channels = planes
        mid_channels = self.norm.adjust_channels_by_factor(mid_channels)
        
        out_channels = planes * self.expansion
        out_channels = self.norm.adjust_channels_by_factor(out_channels)
        
        ## Identity Path
        self.downsample = downsample

        ## Main Path
        self.drop_path = DropPath(drop_prob=self.stochastic_depth_rate)
        self.conv1 = nn.Conv2d(in_planes, mid_channels, 1, stride=1,
                               padding=0, bias=False)
        self.norm1 = self.norm.create(mid_channels)
        self.act1 = self.act.create(inplace=True)
        
        self.conv2 = conv_layer(mid_channels, mid_channels,
                                self.norm, self.act, 
                                kernel_size=3, stride=stride, 
                                groups=groups)
        self.norm2 = self.norm.create(mid_channels)
        self.act2 = self.act.create(inplace=True) 
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, stride=1,
                               padding=0, bias=False)
        self.norm3 = self.norm.create(out_channels)
        
        # Both Paths
        self.act_out = self.act.create(inplace=False)
        
    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)
        
        # residual branch
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x))) 
        x = self.norm3(self.conv3(x))
        x = self.drop_path(x)
        
        out = x + shortcut 
        out = self.act_out(out)
        return out


class Bottle2neck(nn.Module):
    expansion = BLOCK_EXPANSION

    def __init__(
            self, 
            inplanes, 
            planes, 
            norm_factory, 
            act_factory,
            stride=1, 
            downsample=None, 
            groups=1,
            conv_layer=None,
            drop_path_rate=0,
            stochastic_depth=0,
            baseWidth=26, 
            scale=4, 
            ):
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

        self.stochastic_depth_rate = stochastic_depth
        self.drop_path_rate = drop_path_rate
        self.norm = norm_factory
        self.act = act_factory

        conv_layer = conv_layer or nn.Conv2d
        stype = 'stage' if downsample is not None else 'normal'
        
        self.drop_path = DropPath(drop_prob=self.stochastic_depth_rate)
        
        width = int(math.floor(planes * (baseWidth / 64.0)))
        width = self.norm.adjust_channels_by_factor(width)
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = self.norm.create(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv_layer(width, width, self.norm, self.act,
                                    kernel_size=3, stride=stride, 
                                    padding=1, bias=False, groups=groups))
            bns.append(self.norm.create(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = self.norm.create(planes * self.expansion)

        self.relu_outplace = self.act.create(inplace=False)
        self.relu = self.act.create(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        # Main Path
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
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype=='stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.drop_path(out)

        # Identity Path
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        # Final Out
        out += residual
        # out = self.relu(out)
        out = self.relu_outplace(out)

        return out


class BottleneckSE(nn.Module):
    expansion = BLOCK_EXPANSION

    def __init__(
            self, 
            inplanes, 
            planes, 
            norm_factory,
            act_factory,
            stride=1, 
            conv_layer=None,
            downsample=None,
            stochastic_depth=0.
            ):
        
        super().__init__()
        
        self.inplanes = inplanes 
        self.planes = planes
        self.stochastic_depth_rate = stochastic_depth
        self.norm = norm_factory 
        self.act = act_factory
        
        self.drop_path = DropPath(drop_prob=self.stochastic_depth_rate)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm.create(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm.create(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.norm.create(planes * 4)
        
        self.relu = self.act.create(inplace=True)
        
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes * 4, planes // 4, kernel_size=1, bias=True)
        self.conv_up = nn.Conv2d(self.planes // 4, 
                                 self.planes * 4, 
                                 kernel_size=1)
        self.sig = nn.Sigmoid()
        
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = self.drop_path(out1 * out) + residual
        res = self.relu(res)

        return res
    


# ========================================================================== #
# * ### * ### * ### *            Main Network            * ### * ### * ### * #
# ========================================================================== #



class ResNet2d(BaseModel):
    
    stage_dims = STAGE_DIMS
    
    def __init__(self, 
                 block, 
                 layers, 
                 norm_factory,
                 act_factory,
                 conv_layer=None, 
                 baseWidth=26, 
                 in_channels=1, 
                 out_channels=2,              # unused, for compat
                 dropout=0,
                 stochastic_depth=0,
                 reduce_conv1_dstride=False,  # unused, for compat
                 deep_sup=False               # unused, for compat
                 ):
        super().__init__()
        
        # self.stochastic_depth_rate = stochastic_depth
        # self.dropout_rate = dropout
        # self.norm = norm_factory
        # self.act = act_factory
        
        self.inplanes = 64
        # self.baseWidth = baseWidth
        conv_layer = conv_layer or Conv2d
        
        self.set_setting('in_channels', in_channels)
        self.set_setting('out_channels', out_channels)
        self.set_setting('baseWidth', baseWidth)
        self.set_setting('conv_layer', conv_layer)
        self.set_setting('block', block)
        
        self.set_setting('stochastic_depth_rate', stochastic_depth)
        self.set_setting('dropout_rate', dropout) 
        self.set_setting('norm', norm_factory)
        self.set_setting('act', act_factory)
        self.set_setting('deep_sup', deep_sup)
        
        self.set_setting('layers', layers)
        self.set_setting('block_expansion', BLOCK_EXPANSION)
        self.set_setting('stage_dims', STAGE_DIMS)
        
        block.expansion = self.block_expansion
        
        ## Stem
        self.stem = nn.Sequential(
            conv_layer(in_channels, 32, self.norm, self.act, bias=False,
                       kernel_size=3, stride=DOWN_STRIDES, padding=1),
            self.norm.create(32),
            self.act.create(inplace=True),
            conv_layer(32, 32, self.norm, self.act, 
                       kernel_size=3, stride=1, padding=1, bias=False),
            self.norm.create(32),
            self.act.create(inplace=True),
            conv_layer(32, 64, self.norm, self.act, kernel_size=3, 
                       stride=1, padding=1, bias=False),
            self.norm.create(64),
            self.act.create(inplace=False)
        )
        self.stem_down = nn.MaxPool2d(kernel_size=3, 
                                      stride=DOWN_STRIDES, 
                                      padding=1)
        
        ## Stages
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       conv_layer=conv_layer)
        self.layer2 = self._make_layer(block, self.stage_dims[1], layers[1], 
                                       stride=DOWN_STRIDES,
                                       conv_layer=conv_layer)
        self.layer3 = self._make_layer(block, self.stage_dims[2], layers[2], 
                                       stride=DOWN_STRIDES,
                                       conv_layer=conv_layer)
        self.layer4 = self._make_layer(block, self.stage_dims[3], layers[3], 
                                       stride=DOWN_STRIDES,
                                       conv_layer=conv_layer)

        ## Register Hooks for Stage Output Features (before final activation)
        mn = 4  # 2: act-out, 3: bn-out, 4: conv-out
        self.layer0_hook = ForwardHook(list(self.stem.modules())[-mn + 1])
        self.layer1_hook = ForwardHook(list(self.layer1.modules())[-mn])
        self.layer2_hook = ForwardHook(list(self.layer2.modules())[-mn])
        self.layer3_hook = ForwardHook(list(self.layer3.modules())[-mn])
        self.layer4_hook = ForwardHook(list(self.layer4.modules())[-mn])
        
        ## Initialize Parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.print_settings()
    
    def _make_layer(self, block, planes, blocks, conv_layer=None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 or stride == (1, 1, 1):
                pool = nn.Identity() 
            else:
                pool = nn.AvgPool2d(kernel_size=stride, stride=stride, 
                                    ceil_mode=True, count_include_pad=False)
            downsample = nn.Sequential(
                pool,
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                self.norm.create(planes * block.expansion),
            )

        layers = []
        # print(f'Make Layer in_planes {self.inplanes}, planes {planes}')
        layers.append(block(self.inplanes, planes, self.norm, self.act,
                            stride=stride,
                            downsample=downsample, 
                            conv_layer=conv_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # print(f'Make Layer (block {i}) in_planes {self.inplanes}, planes {planes}')
            layers.append(block(self.inplanes, planes, self.norm, self.act,
                                conv_layer=conv_layer))

        return nn.Sequential(*layers)

    def forward(self, x, enc_only=False): 
        # Encoder path
        x2 = self.stem(x)
        x4 = self.stem_down(x2)

        x4  = self.layer1(x4)   # 4x  down, activated (no downsample)
        x8  = self.layer2(x4)   # 8x  down, activated
        x16 = self.layer3(x8)   # 16x down, activated
        x32 = self.layer4(x16)  # 32x down, activated
        
        return {
            '4x':  self.layer1_hook.features,
            '8x':  self.layer2_hook.features,
            '16x': self.layer3_hook.features,
            '32x': self.layer4_hook.features,
            'out': self.layer4_hook.features,
        }
        