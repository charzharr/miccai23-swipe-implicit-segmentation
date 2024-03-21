""" denseunet3d.py (modified version of 2D DenseUNet by: Charley Zhang)
Modified for readability + added modular functionality for densenet
backbones other than 161.

Original DenseUNet 161 implementation:
https://github.com/xmengli999/TCSM/blob/master/models/network.py

DenseUNet-121: ~M params
DenseUNet-169: 47.9M (48.55 deep-sup) params
DenseUNet-201: 59.5M params
DenseUNet-161: ~M params

Modifications List:
    (2021.11.07) Fixed lack of BN + ReLU for UpBlock side-inputs.
        - Added is_sidein_activated & associated modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from . import densenet3d as densenet
from lib.nets.basemodel import BaseModel
from lib.nets.component_factories import NormFactory3d, ActFactory


densenet_outdims = {
    # [7x7+MP, denseblock1, denseblock2, denseblock3, denseblock4 output dims]
    'densenet121': [64, 256, 512, 1024, 1024],
    'densenet169': [64, 256, 512, 1280, 1664],
    'densenet201': [64, 256, 512, 1792, 1920],
    'densenet161': [96, 384, 768, 2112, 2208]
}


class DeepSupBlock(nn.Module):
    def __init__(self, num_upsamples, in_channels, out_channels, 
                 use_deconv=True, is_activated=True):
        super().__init__()
        assert num_upsamples >= 1
        
        self.num_upsamples = num_upsamples
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deconv = use_deconv
        self.is_activated = is_activated
        
        modules_dict = OrderedDict()
        
        if not self.is_activated:
            modules_dict['in_bn'] = nn.BatchNorm3d(in_channels)
            modules_dict['in_act'] = nn.ReLU()
        
        base_channels = max(out_channels, 16)
        for i in range(num_upsamples):
            scale_out_dims = base_channels * 2 ** (self.num_upsamples - i - 1)
            if i == 0:
                modules_dict['in_conv1'] = nn.Conv3d(in_channels, 
                    scale_out_dims, kernel_size=3, padding=1, bias=False)
                modules_dict['in_bn1'] = nn.BatchNorm3d(scale_out_dims)
                modules_dict['in_act1'] = nn.ReLU()
                in_channels = scale_out_dims
            
            if self.use_deconv:
                modules_dict[f'scale{i+1}_up'] = nn.ConvTranspose3d(in_channels, 
                    scale_out_dims, kernel_size=4, stride=2, padding=1, 
                    bias=False)
                modules_dict[f'scale{i+1}_bn'] = nn.BatchNorm3d(scale_out_dims)
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
        


class UpBlock(nn.Module):
    def __init__(self, side_in_dim, bot_in_dim, out_dim, deconv=False,
                 is_sidein_activated=False):
        """
        is_sidein_activated is False if the side input is not BN'd or ReLU'd.
            This is True in Up4 where the side input is the activated 7x7 out.
        """
        super(UpBlock, self).__init__()
        
        self.is_sidein_activated = is_sidein_activated
        if not self.is_sidein_activated:
            self.side_bn = nn.BatchNorm3d(side_in_dim)
            nn.init.constant_(self.side_bn.weight, 1)
            nn.init.constant_(self.side_bn.bias, 0)
        self.dim_mismatched = side_in_dim != bot_in_dim
        if self.dim_mismatched:
            self.match_conv = nn.Conv3d(side_in_dim, bot_in_dim, 1, padding=0,
                                        bias=False)
            nn.init.xavier_normal_(self.match_conv.weight)
            self.match_bn = nn.BatchNorm3d(bot_in_dim)
            nn.init.constant_(self.match_bn.weight, 1)
            nn.init.constant_(self.match_bn.bias, 0)
        
        self.aggregated_conv = nn.Conv3d(bot_in_dim, out_dim, 3, padding=1,
                                         bias=False)
        nn.init.xavier_normal_(self.aggregated_conv.weight)

        self.bn = nn.BatchNorm3d(out_dim)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        self.use_deconv = deconv
        if self.use_deconv:
            self.deconv = nn.ConvTranspose3d(bot_in_dim, bot_in_dim, 4, 
                stride=2, padding=1, bias=False)
            nn.init.xavier_normal_(self.deconv.weight)
        
    def forward(self, side_in, up_in):
        """
        Args:
            side_in (tensor): side input, can be unactivated or activated
            up_in (tensor): activated input from below that needs upsampling
        Returns:
            out (tensor): activated features
        """
        # process input from below (upsample by 2x)
        if self.use_deconv:
            up_in = F.relu(self.deconv(up_in))
        else:
            up_in = F.interpolate(up_in, scale_factor=2, mode='trilinear',
                                  align_corners=True)
        
        # process side input (BN+Act and match feature dims to up_in)
        if not self.is_sidein_activated:
            side_in = F.relu(self.side_bn(side_in))
        if self.dim_mismatched:
            side_in = F.relu(self.match_bn(self.match_conv(side_in)))
        
        # add activated feats and conv
        agg = torch.add(up_in, side_in)   # no cat like U-Net
        out = F.relu(self.bn(self.aggregated_conv(agg)))

        return out


class ForwardHook():
    features = None
    
    def __init__(self, module): 
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): 
        self.features = output

    def remove(self): 
        self.hook.remove()


class DenseUNet(BaseModel):

    def __init__(self, name='densenet161', out_channels=1, deconv=False,
                 norm='batchnorm', act='relu', deep_sup=False):
        super(DenseUNet, self).__init__()
        
        self.features, self.encoder_hooks, ldims = self._setup_encoder(name)
        
        self.up1 = UpBlock(ldims[3], ldims[4], ldims[2], deconv=False, # deconv,
                           is_sidein_activated=False)
        self.up2 = UpBlock(ldims[2], ldims[2], ldims[1], deconv=False, # deconv,
                           is_sidein_activated=False)  
        self.up3 = UpBlock(ldims[1], ldims[1], ldims[0], deconv=False, # deconv,
                           is_sidein_activated=False)
        self.up4 = UpBlock(ldims[0], ldims[0], ldims[0], deconv=False, # deconv,
                           is_sidein_activated=True)

        self.conv1 = nn.Conv3d(ldims[0], 64, kernel_size=3, padding=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        self.use_deconv = deconv
        if self.use_deconv:
            # self.final_deconv = nn.ConvTranspose3d(ldims[0], ldims[0], 3, 
            #     stride=2, padding=1, output_padding=1, bias=False)
            self.final_deconv = nn.ConvTranspose3d(ldims[0], ldims[0], 4, 
                stride=2, padding=1, bias=False)
            nn.init.xavier_normal_(self.final_deconv.weight)
            
        self.deep_sup = deep_sup
        if self.deep_sup:
            self.deepsup_2x = DeepSupBlock(1, ldims[0], out_channels, 
                 use_deconv=True, is_activated=True)
            self.deepsup_4x = DeepSupBlock(2, ldims[0], out_channels, 
                 use_deconv=True, is_activated=True)
            self.deepsup_8x = DeepSupBlock(3, ldims[1], out_channels, 
                 use_deconv=True, is_activated=True)
        
        tot_params, tot_tparams = self.param_counts
        print(f'💠 DenseUNet3d-{name} initialized w/ {out_channels} classes, '
              f'(deep_sup={deep_sup})\n'
              f'   deconv={deconv}, norm={norm}, act={act}, '
                f'{tot_params:,} parameters.')

    def forward(self, x, dropout=False):
        f = F.relu(self.features(x))  # 32x downsampled -> ReLU (is BN'd)

        # Note: UpBlock outputs are all activated (u1, u2, u3, u4)
        u1 = self.up1(self.encoder_hooks[3].features, f)  # cat block3 out -> 16x down
        u2 = self.up2(self.encoder_hooks[2].features, u1)  # cat block2 out -> 8x down
        u3 = self.up3(self.encoder_hooks[1].features, u2)  # cat block1 out -> 4x down
        u4 = self.up4(self.encoder_hooks[0].features, u3)  # cat activated 7x7 out -> 2x

        if self.use_deconv:
            x_fea = F.relu(self.final_deconv(u4))
        else:
            x_fea = F.interpolate(u4, scale_factor=2, mode='trilinear',
                                  align_corners=True)
        x_fea = self.conv1(x_fea)
        
        if dropout:
            x_fea = F.dropout3d(x_fea, p=0.3)
        
        x_fea = F.relu(self.bn1(x_fea))
        x_out = self.conv2(x_fea)
        
        if self.deep_sup:
            return {
                'out': x_out,
                '2x': self.deepsup_2x(u4),
                '4x': self.deepsup_4x(u3),
                '8x': self.deepsup_8x(u2),
            }
        return {
            'out': x_out
        }

    def close(self):
        for hook in self.encoder_hooks: 
            hook.remove()

    def _setup_encoder(self, name):
        if '121' in name:
            base_model = densenet.get_model(
                121, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet121']
        elif '169' in name:
            base_model = densenet.get_model(
                169, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet169']
        elif '201' in name:
            base_model = densenet.get_model(
                201, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet201']
        elif '161' in name:
            base_model = densenet.get_model(
                161, n_input_channels=1, num_classes=1,
                conv1_t_size=7, conv1_t_stride=2)
            layer_dims = densenet_outdims['densenet161']
        else:
            raise Exception(f"Invalid DenseNet name: {name}.")

        # layers is a list of 2 modules: dnet.features, dnet.classifier
        comps = list(base_model.children())
        layers = nn.Sequential(*comps)  # 0: feat extractor, 1: lin classifier
        
        encoder_hooks = [
            ForwardHook(layers[0][2]),  # ReLU after 7x7
            ForwardHook(layers[0][4]),  # DenseBlock 1 out (conv'd)
            ForwardHook(layers[0][6]),  # DenseBlock 2 out (conv'd)
            ForwardHook(layers[0][8]),  # DenseBlock 3 out (conv'd)
        ]
        return layers[0], encoder_hooks, layer_dims


def get_model(name, num_classes=1, deconv=False, deep_sup=False,
              norm='batchnorm', act='relu'):
    model = DenseUNet(str(name), out_channels=num_classes, deconv=deconv,
                      deep_sup=deep_sup, norm=norm, act=act)
    return model
