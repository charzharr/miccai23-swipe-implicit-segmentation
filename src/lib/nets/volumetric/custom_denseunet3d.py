""" denseunet3d.py (modified version of 2D DenseUNet by: Charley Zhang)
Modified for readability + added modular functionality for densenet
backbones other than 161.

Original DenseUNet 161 implementation:
https://github.com/xmengli999/TCSM/blob/master/models/network.py

DenseUNet-121: ~M params
DenseUNet-169: ~M params
DenseUNet-201: 59.5M params
DenseUNet-161: ~M params
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

if __name__ == '__main__':
    import sys, pathlib
    curr_path = pathlib.Path(__file__).parent.absolute()
    sys.path.append(str(curr_path.parent.parent.parent))
from lib.nets.basemodel import BaseModel


def get_model(model_depth, **kwargs):
    """
    To get densenet model. NOT FOR CUSTOM DENSEUNET.
    """
    model_depth = int(model_depth)
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)
    return model


class ForwardHook:
    def __init__(self, module):
        self.activations = None
        self.forward_hook = module.register_forward_hook(self.forward_hook_fn)
        
    def forward_hook_fn(self, module, input, output):
        self.activations = output


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, stage_index, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        if stage_index != 0:
            self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(BaseModel):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first 
            convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super().__init__()

        self.hooks = []

        # First convolution
        first_module = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(n_input_channels,
                                num_init_features,
                                kernel_size=(conv1_t_size, 7, 7),
                                stride=(conv1_t_stride, 2, 2),
                                padding=(conv1_t_size // 2, 3, 3),
                                bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True))]))
        self.hooks.append(ForwardHook(first_module))
        self.features = first_module
        if not no_max_pool:
            self.features.add_module(
                'pool1', nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(i, num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

                if i in (0, 1):
                    self.hooks.append(ForwardHook(trans))

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out



### ---------------------------  ASPP Module  --------------------------- ###


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, 
                                     dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes, dilations=[1, 2, 4, 8]):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(256, 256, 1, padding=0, 
                        dilation=dilations[0])
        self.aspp2 = _ASPPModule(256, 256, 3, padding=dilations[1], 
                        dilation=dilations[1])
        self.aspp3 = _ASPPModule(256, 256, 3, padding=dilations[2], 
                        dilation=dilations[2])
        self.aspp4 = _ASPPModule(256, 256, 3, padding=dilations[3], 
                        dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.MaxPool3d(3, stride=1, padding=1),
                                             nn.Conv3d(256, 256, 1, 
                                                       stride=1, bias=False),
                                             nn.BatchNorm3d(256),
                                             nn.ReLU(inplace=True))
        self.conv0 = nn.Conv3d(inplanes, 256, 1, bias=False)
        self.bn0 = nn.BatchNorm3d(256)
        self.conv1 = nn.Conv3d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.dropout(x)
        return x


### -------------------------  DenseUNet Module  ------------------------- ###



densenet_outdims = {
    # [7x7+MP, denseblock1, denseblock2, denseblock3, denseblock4 output dims]
    'densenet121': [64, 256, 512, 1024, 1024],
    'densenet169': [64, 256, 512, 1280, 1664],
    'densenet201': [64, 256, 512, 1792, 1920],
    'densenet161': [96, 384, 768, 2112, 2208]
}


class UpBlock(nn.Module):
    def __init__(self, side_in_dim, bot_in_dim, out_dim, deconv=False):
        super(UpBlock, self).__init__()

        self.use_deconv = deconv
        if self.use_deconv:
            self.deconv = nn.ConvTranspose3d(bot_in_dim, bot_in_dim,
                                             kernel_size=2, stride=2)
            nn.init.xavier_normal_(self.deconv.weight)
        
        self.side_conv = nn.Conv3d(side_in_dim, bot_in_dim, 1, padding=0)
        nn.init.xavier_normal_(self.side_conv.weight)
        
        self.aggregated_conv = nn.Conv3d(bot_in_dim, out_dim, 3, padding=1)
        nn.init.xavier_normal_(self.aggregated_conv.weight)

        self.bn = nn.BatchNorm3d(out_dim)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, side_in, bot_in):
        if self.use_deconv:
            bot_in = self.deconv(bot_in)
        else:
            bot_in = F.interpolate(bot_in, scale_factor=2, mode='trilinear', 
                align_corners=True)
        
        side_in = self.side_conv(side_in)
        
        agg = torch.add(bot_in, side_in)   # no cat like U-Net
        out = F.relu(self.bn(self.aggregated_conv(agg)))

        return out


class DenseUNet(BaseModel):

    def __init__(self, name='densenet161', out_channels=1, deconv=True):
        super(DenseUNet, self).__init__()
        
        self.features, self.encoder_hooks, ldims = self._setup_encoder(name)
        self.aspp = ASPP(ldims[-1])  # 256 out
        
        # UpBlock(side_in, bot_in, out_dim)
        self.up1 = UpBlock(256, 256, 128, deconv=True)
        self.up2 = UpBlock(128, 128, 64, deconv=True)  
        self.up3 = UpBlock(64, 64, 64, deconv=True)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)
        self.final_up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', 
                                    align_corners=True)
        
        nn.init.xavier_normal_(self.final_conv.weight)

        tot_params, tot_tparams = self.param_counts
        print(f'💠 CustomDenseUNet3d-{name} initialized w/ {out_channels} classes,\n'
              f'   deconv={deconv}, and {tot_params:,} parameters.')

    def forward(self, x, enc_only=False):
        f = self.features(x)  # down by 8x, 16x, 16x -> ReLU (is BN'd)
        if enc_only:
            return {'out': f}

        f = F.relu(f)
        f = self.aspp(f)

        x = self.up1(self.encoder_hooks[2].activations, f)  
        x = self.up2(self.encoder_hooks[1].activations, x)  
        x = self.up3(self.encoder_hooks[0].activations, x)  
        
        x = self.final_conv(x)
        x = self.final_up(x)
        return {'out': x, 'features': f}

        """ Old code
        x = self.up1(self.encoder_hooks[3].features, f)  # cat block3 out -> 16x down
        x = self.up2(self.encoder_hooks[2].features, x)  # cat block2 out -> 8x down
        x = self.up3(self.encoder_hooks[1].features, x)  # cat block1 out -> 4x down
        x = self.up4(self.encoder_hooks[0].features, x)  # cat activated 7x7 out -> 2x
        """

    def close(self):
        for hook in self.encoder_hooks: 
            hook.remove()

    def _setup_encoder(self, name):
        if '121' in name:
            base_model = get_model(
                121, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet121']
        elif '169' in name:
            base_model = get_model(
                169, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet169']
        elif '201' in name:
            base_model = get_model(
                201, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet201']
        elif '161' in name:
            base_model = get_model(
                161, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet161']
        else:
            raise Exception(f"Invalid DenseNet name: {name}.")

        # layers is a list of 2 modules: dnet.features, dnet.classifier
        comps = list(base_model.children())
        layers = nn.Sequential(*comps)  # 0: feat extractor, 1: lin classifier
        
        encoder_hooks = [
            ForwardHook(layers[0].relu1),  
            ForwardHook(layers[0].transition1.conv),  
            ForwardHook(layers[0].transition2.pool)
        ]
        return layers[0], encoder_hooks, layer_dims


if __name__ == '__main__':
    model = DenseUNet(name='201')
    import IPython; IPython.embed(); 
