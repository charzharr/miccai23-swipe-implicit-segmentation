"""  Module nets/volumetric/cumednet3d (Author: Charley Zhang)
3D version of CUMedNet. 

Pengfei's truncated model had the following config:
    nc -> 2nc, 2nc -> 3nc (2), 3nc -> 6nc (2), 6nc -> 12nc (2), 12nc -> 12nc (2)

    CUMedNet3d(1, 14, 
        block_module=BasicBlock,
        stage_counts=[2, 2, 2, 2, 2], 
        stage_expansions=[2, 3, 6, 12, 12],
        init_features=64,
        concat_features=16)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


norm = nn.BatchNorm3d
use_prenorm_bias = False  # False for BN
activation = nn.ReLU(inplace=True)

resnet_stages = {  # Note: these assume 1 7x7 initial conv & 1 linear layer
    10: [1, 1, 1, 1],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],  # BasicBlock 16(2) + 2
    50: [3, 4, 6, 3],  # Bottleneck 16(3) + 2
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}


class TransConv_(nn.Module):
    """4x4 deconvolution with padding"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.tconv = nn.ConvTranspose3d(in_features, out_features, 
                        kernel_size=4, stride=2, padding=1)
        self.bn = norm(out_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeconvStage(nn.Module):
    def __init__(self, in_features, out_features, down_factor):
        """
        Args:
            down_factor: how downscaled an input is
        """
        super().__init__()

        layer_in_feats = in_features
        layer_out_feats = out_features * 2 ** (down_factor - 2)

        layers = []
        for i in range(down_factor - 1):
            if i == down_factor - 2:
                assert layer_out_feats == out_features
            layers.append(TransConv_(layer_in_feats, layer_out_feats))
            layer_in_feats = layer_out_feats
            layer_out_feats = math.floor(layer_out_feats / 2)
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out



class BottleneckBlock(nn.Module):
    """ Input is always downsampled by 2x via the first 3x3 conv layer. """
    shrinkage = 4

    def __init__(self, in_features, out_features):
        super().__init__()
        
        if in_features != out_features:  # not first block in stage
            self.residual_branch = nn.Sequential(
                nn.AvgPool3d(2, stride=2),
                nn.Conv3d(in_features, out_features, kernel_size=1, stride=1,
                          bias=use_prenorm_bias),
                norm(out_features)
            )
        else:
            self.residual_branch = None

        intermediate_features = out_features // self.shrinkage
        stride = 2 if in_features != out_features else 1
        self.main_branch = nn.Sequential(
            nn.Conv3d(in_features, intermediate_features, kernel_size=1, 
                      stride=1, padding=0, bias=use_prenorm_bias),
            norm(intermediate_features),
            activation,
            nn.Conv3d(intermediate_features, intermediate_features, 
                     kernel_size=3, stride=stride, padding=1, bias=use_prenorm_bias),
            norm(intermediate_features),
            activation,
            nn.Conv3d(intermediate_features, out_features, kernel_size=1, 
                      stride=1, padding=0, bias=use_prenorm_bias),
            norm(out_features),
        )
        
    def forward(self, x):
        residual = x

        out = self.main_branch(x)

        if self.residual_branch is not None:
            residual = self.residual_branch(residual)

        out += residual
        out = activation(out)
        return out


class BasicBlock(nn.Module):
    """ Input is always downsampled by 2x via the first 3x3 conv layer. """

    def __init__(self, in_features, out_features):
        super().__init__()
        
        if in_features != out_features:  # not first block in stage
            self.residual_branch = nn.Sequential(
                nn.AvgPool3d(2, stride=2),
                nn.Conv3d(in_features, out_features, kernel_size=1, stride=1,
                          bias=use_prenorm_bias),
                norm(out_features)
            )

        else:
            self.residual_branch = None

        stride = 2 if in_features != out_features else 1
        self.main_branch = nn.Sequential(
            nn.Conv3d(in_features, out_features, kernel_size=3, 
                      stride=stride, padding=1, bias=use_prenorm_bias),
            norm(out_features),
            activation,
            nn.Conv3d(out_features, out_features, kernel_size=3, 
                      stride=1, padding=1, bias=use_prenorm_bias),
            norm(out_features)
        )
        
    def forward(self, x):
        residual = x

        out = self.main_branch(x)

        if self.residual_branch is not None:
            residual = self.residual_branch(residual)

        out += residual
        out = activation(out)
        return out


class Stage(nn.Module):
    def __init__(self, in_features, out_features, num_blocks, block_module):
        super().__init__()
        self.num_blocks = num_blocks
        
        layers_list = []
        for i in range(self.num_blocks):
            if i == 0:
                layers_list.append(block_module(in_features, out_features))
            else:
                layers_list.append(block_module(out_features, out_features))
        self.stage_modules = nn.Sequential(*layers_list)
    
    def forward(self, x):
        out = self.stage_modules(x)
        return out


class CUMedNet3d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            num_classes, 
            stage_counts=[3, 4, 6, 3],  # defaults to ResNet-50
            block_module=BottleneckBlock,
            init_channels=64,
            stage_expansions=[2, 4, 8, 16],
            concat_channels=16,
            use_add=True,
            deep_sup=False):
        """
        Args:
            in_channels: num of input channels
            num_classes: num of output channels output layer(s)
            init_channels: num of initial channels
            concat_channels: num of channels to output before final concat
            use_add: flag to add channels, if false then concatenation is used
        """
        assert len(stage_counts) == len(stage_expansions)
        super().__init__()

        self.use_add = use_add
        self.num_classes = num_classes
        self.deep_sup = deep_sup

        self.scale1 = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size=3, 
                      stride=1, padding=1, bias=use_prenorm_bias),
            norm(init_channels),
            activation,
            nn.Conv3d(init_channels, init_channels, kernel_size=3, 
                      stride=1, padding=1, bias=use_prenorm_bias),
            norm(init_channels),
            activation
        )
        self.tconv_stage1 = nn.Sequential(
            nn.Conv3d(init_channels, concat_channels, kernel_size=3, 
                      stride=1, padding=1, bias=use_prenorm_bias),
            norm(concat_channels),
            activation
        )

        # Create Stages and UpResolution Moduels
        up_modules = []
        stages = []
        stage_in_channels = init_channels
        for i, n_blocks in enumerate(stage_counts):
            stage_out_channels = int(stage_expansions[i] * init_channels)

            stages.append(
                Stage(stage_in_channels, 
                      stage_out_channels,
                      n_blocks,
                      block_module)
            )
            up_modules.append(
                DeconvStage(stage_out_channels, concat_channels, i + 2)
            )
            stage_in_channels = stage_out_channels
        self.stages = nn.ModuleList(stages)
        self.tconv_stages = nn.ModuleList(up_modules)

        #final conv 3*3
        if self.use_add:
            self.final_conv1 = nn.Conv3d(concat_channels, concat_channels, 
                                         kernel_size=3, stride=1, padding=1,
                                         bias=use_prenorm_bias)
        else:
            self.final_conv1 = nn.Conv3d(concat_channels * (len(stages) + 1), 
                                         concat_channels, 
                                         kernel_size=3, stride=1, padding=1,
                                         bias=use_prenorm_bias)
        self.final_norm1 = norm(concat_channels)
        self.final_conv2 = nn.Conv3d(concat_channels, num_classes,
                                     kernel_size=1, stride=1, padding=0)
        
        #initialize_weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
       
  
    def forward(self, x, enc_only=False):
        s1 = self.scale1(x)
        x = s1
        # print('s1', x.shape)

        stage_outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # print(f's{i+1}', x.shape)
            stage_outputs.append(x)

        upsampled_outputs = [self.tconv_stage1(s1)]
        for stage_out, tconv_stage in zip(stage_outputs, self.tconv_stages):
            upsampled_outputs.append(tconv_stage(stage_out))

        if self.use_add:
            accumulated = upsampled_outputs[0]
            for i in range(1, len(upsampled_outputs)):
                accumulated = accumulated + upsampled_outputs[i]
        else:
            accumulated = torch.cat(upsampled_outputs, dim=1)

        out = activation(self.final_norm1(self.final_conv1(accumulated)))
        out = self.final_conv2(out)

        return {'out': out}


if __name__ == '__main__':
    model = CUMedNet3d( 
                1, 
                14, 
                stage_counts=[2, 4, 6, 2],  # defaults to ResNet-50
                block_module=BottleneckBlock,
                init_channels=64,
                stage_expansions=[2, 4, 8, 16],
                concat_channels=16,
                use_add=False,
                deep_sup=False).cuda()

    X = torch.randn((1, 1, 64, 64, 64)).cuda()
    o = model(X)
    print(o['out'].shape)
    import IPython; IPython.embed(); 


