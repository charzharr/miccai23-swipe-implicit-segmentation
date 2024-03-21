""" Module model_backbone_v2.py

> Motivation for Changes:
Revisiting 3D ResNets for Video Recog. (https://arxiv.org/pdf/2109.01696v1.pdf)
    - Training regularization yields performance increases. 
    - SlowFast seems to perform much better than (2+1)D.
    - My interpretation of PreActivation seems to be incorrect.
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

from lib.nets.basemodel import BaseModel
from lib.nets.component_factories import NormFactory3d, ActFactory
from lib.nets.modules.droppath import DropPath


DOWN_STRIDES = (2, 2, 2)
BLOCK_EXPANSION = 4                # orig: 4
STAGE_DIMS = [64, 128, 256, 512]   # orig: [64, 128, 256, 512], 1st must =64
                                   #  [64, 96, 128, 160]



# ========================================================================== #
# * ### * ### * ### *            Layer Modules           * ### * ### * ### * #
# ========================================================================== #


class Conv3d(nn.Module):
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
        
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        
    def forward(self, x):
        x = self.conv(x)
        return x


class Conv2p1(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            norm_factory,
            act_factory,
            kernel_size=3, 
            stride=1, 
            padding=1,
            groups=1, 
            bias=False
            ):
        super().__init__()
        self.norm = norm_factory
        self.act = act_factory
        
        self.stride = stride 
        self.padding = padding
        self.kernel_size = kernel_size
        
        inter_channels = self._mid_channels(kernel_size, 
                                            in_channels, 
                                            out_channels)
        inter_channels = self.norm.adjust_channels_by_factor(inter_channels)
        
        ## Spatial -> Temporal Conv Layers
        self.conv_spatial = nn.Conv3d(in_channels,
                                      inter_channels,
                                      kernel_size=self._kernel_size(spatial=True),
                                      stride=self._stride(spatial=True),
                                      padding=self._padding(spatial=True),
                                      groups=groups,
                                      bias=False)
        self.bn_spatial = self.norm.create(inter_channels)
        self.act_spatial = self.act.create(inplace=True)

        self.conv_temporal = nn.Conv3d(inter_channels,
                                       out_channels,
                                       kernel_size=self._kernel_size(spatial=False),
                                       stride=self._stride(spatial=False),
                                       padding=self._padding(spatial=False),
                                       groups=groups,
                                       bias=bias)
        
    def forward(self, x):
        x = self.conv_spatial(x)
        x = self.act_spatial(self.bn_spatial(x))
        x = self.conv_temporal(x)
        return x
    
    def _kernel_size(self, spatial=True):
        value = self.kernel_size
        if spatial:
            if isinstance(value, Sequence):
                assert len(value) == 3
                return (1, value[1], value[2])
            return (1, value, value)
        if isinstance(value, Sequence):
            assert len(value) == 3
            return (value[0], 1, 1)
        return (value, 1, 1)
    
    def _stride(self, spatial=True):
        if spatial:
            if isinstance(self.stride, Sequence):
                assert len(self.stride) == 3
                return (1, self.stride[1], self.stride[2])
            return (1, self.stride, self.stride)
        if isinstance(self.stride, Sequence):
            assert len(self.stride) == 3
            return (self.stride[0], 1, 1)
        return (self.stride, 1, 1)
    
    def _padding(self, spatial=True):
        value = self.padding
        if spatial:
            if isinstance(value, Sequence):
                assert len(value) == 3
                return (0, value[1], value[2])
            return (0, value, value)
        if isinstance(value, Sequence):
            assert len(value) == 3
            return (value[0], 0, 0)
        return (value, 0, 0)
    
    def _mid_channels(self, kernel_size, in_channels, out_channels):
        """ M_i = floor([t * d^2 N_i-1 * N_i] / [ d^2 * N_i-1 + t * N_i ]) 
            d: spatial kernel size, t: temporal kernel size, N: in/out planes """
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * 3
        
        mid_channels = kernel_size[0] * kernel_size[1] * kernel_size[2] * \
                       in_channels * out_channels
        mid_channels /= (in_channels * kernel_size[1] * kernel_size[2] + \
                         kernel_size[0] * out_channels)
        mid_channels = round(mid_channels) // 2  # added for lower mem
        return mid_channels



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
        
        conv_layer = conv_layer or Conv3d
        mid_channels = planes
        mid_channels = self.norm.adjust_channels_by_factor(mid_channels)
        
        out_channels = planes * self.expansion
        out_channels = self.norm.adjust_channels_by_factor(out_channels)
        
        ## Identity Path
        self.downsample = downsample

        ## Main Path
        self.drop_path = DropPath(drop_prob=self.stochastic_depth_rate)
        self.conv1 = nn.Conv3d(in_planes, mid_channels, 1, stride=1,
                               padding=0, bias=False)
        self.norm1 = self.norm.create(mid_channels)
        self.act1 = self.act.create(inplace=True)
        
        self.conv2 = conv_layer(mid_channels, mid_channels,
                                self.norm, self.act, 
                                kernel_size=3, stride=stride, 
                                groups=groups)
        self.norm2 = self.norm.create(mid_channels)
        self.act2 = self.act.create(inplace=True) 
        
        self.conv3 = nn.Conv3d(mid_channels, out_channels, 1, stride=1,
                               padding=0, bias=False)
        self.norm3 = self.norm.create(out_channels)
        
        # Both Paths
        self.act_out = self.act.create(inplace=True)
        
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

        conv_layer = conv_layer or nn.Conv3d
        stype = 'stage' if downsample is not None else 'normal'
        
        self.drop_path = DropPath(drop_prob=self.stochastic_depth_rate)
        
        width = int(math.floor(planes * (baseWidth / 64.0)))
        width = self.norm.adjust_channels_by_factor(width)
        self.conv1 = nn.Conv3d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = self.norm.create(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool3d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv_layer(width, width, self.norm, self.act,
                                    kernel_size=3, stride=stride, 
                                    padding=1, bias=False, groups=groups))
            bns.append(self.norm.create(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(width*scale, planes * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = self.norm.create(planes * self.expansion)

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
        out = self.relu(out)

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
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm.create(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm.create(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.norm.create(planes * 4)
        
        self.relu = self.act.create(inplace=True)
        
        # SE
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_down = nn.Conv3d(
            planes * 4, planes // 4, kernel_size=1, bias=True)
        self.conv_up = nn.Conv3d(self.planes // 4, 
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


class DeepSupBlock3d(nn.Module):
    def __init__(
            self, 
            num_upsamples, 
            in_channels, 
            out_channels, 
            norm_factory,
            act_factory,
            use_deconv=True):
        
        super().__init__()
        
        self.norm = norm_factory
        self.act = act_factory 
        
        assert num_upsamples >= 1
        
        self.num_upsamples = num_upsamples
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deconv = use_deconv
        
        modules_dict = OrderedDict()
        base_channels = max(out_channels, 16)
        for i in range(num_upsamples):
            scale_out_dims = base_channels * 2 ** (self.num_upsamples - i - 1)
            scale_out_dims = self.norm.adjust_channels_by_factor(scale_out_dims)
            if i == 0:
                modules_dict['in_conv1'] = nn.Conv3d(in_channels, 
                    scale_out_dims, kernel_size=3, padding=1, bias=False)
                modules_dict['in_bn1'] = self.norm.create(scale_out_dims)
                modules_dict['in_act1'] = self.act.create(inplace=True)
                in_channels = scale_out_dims
            
            if self.use_deconv:
                modules_dict[f'scale{i+1}_up'] = nn.ConvTranspose3d(in_channels, 
                    scale_out_dims, kernel_size=4, stride=2, padding=1, 
                    bias=False)
                modules_dict[f'scale{i+1}_bn'] = self.norm.create(scale_out_dims)
                modules_dict[f'scale{i+1}_act'] = self.act.create(inplace=True)
            else:
                modules_dict[f'scale{i+1}_up'] = nn.Upsample(scale_factor=2,
                    mode='bilinear', align_corners=True)
                
            if i == num_upsamples - 1: 
                modules_dict['final_conv'] = nn.Conv3d(scale_out_dims, 
                    out_channels, kernel_size=1, bias=True) 
            in_channels = scale_out_dims
                
        self.decoder = nn.Sequential(modules_dict)
    
    def forward(self, x):
        return self.decoder(x)

    
class UpBlock(nn.Module):
    """
    Upsamples via linear interpolation.
    """
    def __init__(
            self, 
            side_in_dim, 
            bot_in_dim, 
            out_dim, 
            norm_factory, 
            act_factory,
            conv_layer=None,
            ):

        self.norm = norm_factory
        self.act = act_factory
                
        super(UpBlock, self).__init__()
        
        conv_layer = conv_layer or Conv3d
        
        self.side_conv = nn.Conv3d(side_in_dim, out_dim, 1, bias=False)
        self.side_norm = self.norm.create(out_dim)

        self.bot_conv = nn.Conv3d(bot_in_dim, out_dim, 1, bias=False)
        self.bot_norm = self.norm.create(out_dim)
        
        self.side_plus_bot_act = self.act.create(inplace=True)
        
        self.final_conv = conv_layer(out_dim, out_dim, self.norm, self.act,
                                     kernel_size=3, padding=1, 
                                     stride=1, bias=False)
        self.final_norm = self.norm.create(out_dim)
        self.final_act = self.act.create(inplace=True)
        
    def forward(self, side_in, bot_in):
        """
        Args:
            side_in (tensor): side input
            up_in (tensor): activated input from below that needs upsampling
        Returns:
            out (tensor): activated features
        """
        # Bottom Input from below (upsample by 2x)
        bot = F.interpolate(bot_in, scale_factor=DOWN_STRIDES, mode='trilinear',
                            align_corners=True)
        bot = self.bot_norm(self.bot_conv(bot))
        
        # Side Input
        side = self.side_norm(self.side_conv(side_in))
        
        # Aggregation + Output
        agg = self.side_plus_bot_act(bot + side)
        out = self.final_act(self.final_norm(self.final_conv(agg)))
        
        return out
    

# ========================================================================== #
# * ### * ### * ### *            Main Network            * ### * ### * ### * #
# ========================================================================== #    

    
class ResUNet3d(BaseModel):
    
    stage_dims = STAGE_DIMS
    
    def __init__(self, 
                 block, 
                 layers, 
                 norm_factory,
                 act_factory,
                 conv_layer=None, 
                 baseWidth=26, 
                 in_channels=1, 
                 out_channels=1, 
                 dropout=0,
                 stochastic_depth=0,
                 reduce_conv1_dstride=False,
                 deep_sup=False
                 ):
        super().__init__()
        
        # self.stochastic_depth_rate = stochastic_depth
        # self.dropout_rate = dropout
        # self.norm = norm_factory
        # self.act = act_factory
        
        self.inplanes = 64
        # self.baseWidth = baseWidth
        conv_layer = conv_layer or Conv3d
        
        self.set_setting('in_channels', in_channels)
        self.set_setting('out_channels', out_channels)
        self.set_setting('baseWidth', baseWidth)
        self.set_setting('conv_layer', conv_layer)
        self.set_setting('block', block)
        self.set_setting('reduce_conv1_dstride', reduce_conv1_dstride)
        self.set_setting('deep_sup', deep_sup)
        
        self.set_setting('stochastic_depth_rate', stochastic_depth)
        self.set_setting('dropout_rate', dropout) 
        self.set_setting('norm', norm_factory)
        self.set_setting('act', act_factory)
        
        self.set_setting('layers', layers)
        self.set_setting('block_expansion', BLOCK_EXPANSION)
        self.set_setting('stage_dims', STAGE_DIMS)
        
        block.expansion = self.block_expansion
        
        ## Stem
        conv1_stride = (1, 2, 2) if self.reduce_conv1_dstride else DOWN_STRIDES
        self.stem = nn.Sequential(
            conv_layer(in_channels, 32, self.norm, self.act, bias=False,
                       kernel_size=3, stride=conv1_stride, padding=1),
            self.norm.create(32),
            self.act.create(inplace=True),
            conv_layer(32, 32, self.norm, self.act, 
                       kernel_size=3, stride=1, padding=1, bias=False),
            self.norm.create(32),
            self.act.create(inplace=True),
            conv_layer(32, 64, self.norm, self.act, kernel_size=3, 
                       stride=1, padding=1, bias=False),
            self.norm.create(64),
            self.act.create(inplace=True)
        )
        self.stem_down = nn.MaxPool3d(kernel_size=3, 
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
        
        ## Decoder 
        self.up16 = UpBlock(1024, 2048, 256, self.norm, self.act,
                            conv_layer=conv_layer)
        self.up8  = UpBlock(512, 256, 128, self.norm, self.act,
                            conv_layer=conv_layer)
        self.up4  = UpBlock(256, 128, 64, self.norm, self.act,
                            conv_layer=conv_layer)
        self.up2  = UpBlock(64, 64, 64, self.norm, self.act,
                            conv_layer=conv_layer)

        ## Final Connector to Prediction Heads
        self.up2_conv = conv_layer(64, 32, self.norm, self.act, bias=False,
                                   kernel_size=3, padding=1, stride=1)
        self.up2_norm = self.norm.create(32)
        self.up2_act = self.act.create(inplace=True)
        self.final_conv = nn.Conv3d(32, self.out_channels, 1, bias=True)
        
        if self.deep_sup:
            self.up2_deepsup = DeepSupBlock3d(1, 64, self.out_channels,
                                              self.norm, self.act)
            self.up4_deepsup = DeepSupBlock3d(2, 64, self.out_channels,
                                              self.norm, self.act)
            self.up8_deepsup = DeepSupBlock3d(3, 128, self.out_channels,
                                              self.norm, self.act)
        
        self.dropout = None 
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout3d(p=self.dropout_rate)

        ## Initialize Parameters
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.print_settings()
    
    def _make_layer(self, block, planes, blocks, conv_layer=None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 or stride == (1, 1, 1):
                pool = nn.Identity() 
            else:
                pool = nn.AvgPool3d(kernel_size=stride, stride=stride, 
                                    ceil_mode=True, count_include_pad=False)
            downsample = nn.Sequential(
                pool,
                nn.Conv3d(self.inplanes, planes * block.expansion, 
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
        # return nn.ModuleList(layers)
        return nn.Sequential(*layers)

    def forward(self, x, enc_only=False): 
        # Encoder path
        x2 = self.stem(x)
        x4 = self.stem_down(x2)

        x4  = self.layer1(x4)   # 4x  down, activated (no downsample)
        x8  = self.layer2(x4)   # 8x  down, activated
        x16 = self.layer3(x8)   # 16x down, activated
        x32 = self.layer4(x16)  # 32x down, activated

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
        out8  = self.up8(x8, out16)
        out4  = self.up4(x4, out8)
        out2  = self.up2(x2, out4)

        out = self.up2_norm(self.up2_conv(out2))
        
        scale_factor = (1, 2, 2) if self.reduce_conv1_dstride else 2
        out = F.interpolate(out, scale_factor=scale_factor, 
                            mode='trilinear', align_corners=True)
        out = self.up2_act(out)
        out = self.final_conv(out)
                
        if self.dropout is not None:
            out = self.dropout(out)

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
                '32x': x32,
                'dec_16x': out16,
                'dec_8x': out8,
                'dec_4x': out4,
                'dec_2x': out2
            }