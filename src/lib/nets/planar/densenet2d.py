""" densenet.py (modified by: Charley Zhang 2020.08)
Modified implementation of PyTorch densenet from:
https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

Model Overviews  (7 output classes)
-------------
DenseNet121: 6.9M params  | 74.43% Top1
DenseNet169: 12.5M params | 75.60% Top1
DenseNet201: 18.1M params | 76.90% Top1
DenseNet161: 26.5M params | 77.14% Top1
"""

import sys, os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..basemodel import BaseModel

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_params_path = '/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Models/'
model_params = {
    'densenet121': os.path.join(model_params_path, 'densenet121.pth'),
    'densenet169': os.path.join(model_params_path, 'densenet169.pth'),
    'densenet201': os.path.join(model_params_path, 'densenet201.pth'),
    'densenet161': os.path.join(model_params_path, 'densenet161.pth'),
}

def _get_params(model_name):
    assert model_name in model_params, f"{model_name} not in params list."
    return torch.load(model_params[model_name])


class ForwardHook:
    def __init__(self, module):
        self.activations = None
        self.forward_hook = module.register_forward_hook(self.forward_hook_fn)
        
    def forward_hook_fn(self, module, input, output):
        self.activations = output


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, 
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, 
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(BaseModel):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.2, 
                 num_classes=1000, final_drop_rate=0.):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, 
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        self.hooks = []
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, 
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            self.hooks.append(ForwardHook(block))
            
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, 
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.final_drop_rate = final_drop_rate

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, activations=False, ret_dict=False):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        if self.final_drop_rate > 0:
            # print('Final drop rate: ', self.final_drop_rate)
            out = F.dropout(out, p=self.final_drop_rate, 
                            training=self.training)
        activs = out
        out = self.classifier(out)
        
        if ret_dict:
            return {
                'features': features,
                'activations': activs,
                'out': out
            }
        
        if activations:  # backward compat.
            return out, activs
        else:
            return out


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Structure:
        Bx64xH/4xW/4 @ 7x7+MP out
        Bx256xH/8xW/8 @ denseblock1 out (before transition //2 dims)
        Bx512xH/16xW/16 @ denseblock 2 out
        Bx1024xH/32xW/32 @ denseblock 3 out
        Bx1024xH/32xW/32 @ denseblock 4 out
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    num_classes = kwargs['num_classes']
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = _get_params('densenet121')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            if num_classes != 1000 and 'classifier' in key:
                del state_dict[key]
        load_state_dict(model, state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Structure:
        Bx64xH/4xW/4 @ 7x7+MP out
        Bx256xH/8xW/8 @ denseblock1 out (before transition //2 dims)
        Bx512xH/16xW/16 @ denseblock 2 out
        Bx1280xH/32xW/32 @ denseblock 3 out
        Bx1664xH/32xW/32 @ denseblock 4 out
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    num_classes = kwargs['num_classes']
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = _get_params('densenet169')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            if num_classes != 1000 and 'classifier' in key:
                del state_dict[key]
        load_state_dict(model, state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Structure:
        Bx64xH/4xW/4 @ 7x7+MP out
        Bx256xH/8xW/8 @ denseblock1 out (before transition //2 dims)
        Bx512xH/16xW/16 @ denseblock 2 out
        Bx1792xH/32xW/32 @ denseblock 3 out
        Bx1920xH/32xW/32 @ denseblock 4 out
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    num_classes = kwargs['num_classes']
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = _get_params('densenet201')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            if num_classes != 1000 and 'classifier' in key:
                del state_dict[key]
        load_state_dict(model, state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Structure:
        Bx96xH/4xW/4 @ 7x7+MP out
        Bx384xH/8xW/8 @ denseblock1 out (before transition //2 dims)
        Bx768xH/16xW/16 @ denseblock 2 out
        Bx2112xH/32xW/32 @ denseblock 3 out
        Bx2208xH/32xW/32 @ denseblock 4 out
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    num_classes = kwargs['num_classes']
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = _get_params('densenet161')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            if num_classes != 1000 and 'classifier' in key:
                del state_dict[key]
        load_state_dict(model, state_dict)
    return model


def load_state_dict(model, state_dict):
    print(' (DenseNet pretrained=True) Loading pretrained stat_dict..')
    print('\t', model.load_state_dict(state_dict, strict=False))


def get_model(name, pretrained=True, only_encoder=False, layer_drop_rate=0.2, 
              final_drop_rate=0., num_classes=1000):
    """
    Parameters (defaults are based on PyTorch values):
        pretrained (bool) - use ImageNet pretrained parameters
        only_encoder (bool) - only get feature extractor (full FMs)
        layer_drop_rate (probability) - drop rate after every dense layer
        final_drop_rate (probability) - drop rate before last pooling layer
        num_classes (int) - number of output classes (cut if only_encoder)
    """
    name = str(name)
    if '121' in name:
        model = densenet121(pretrained=pretrained, num_classes=num_classes,
                    drop_rate=layer_drop_rate, final_drop_rate=final_drop_rate)
    elif '169' in name:
        model = densenet169(pretrained=pretrained, num_classes=num_classes,
                    drop_rate=layer_drop_rate, final_drop_rate=final_drop_rate)
    elif '201' in name:
        model = densenet201(pretrained=pretrained, num_classes=num_classes,
                    drop_rate=layer_drop_rate, final_drop_rate=final_drop_rate)
    elif '161' in name:
        model = densenet161(pretrained=pretrained, num_classes=num_classes,
                    drop_rate=layer_drop_rate, final_drop_rate=final_drop_rate)
    else:
        raise ValueError(f"DenseNet name ({name}) is not supported.")
    
    if only_encoder:
        model = list(model.children())[0]

    print(f" (DenseNet get_model) {name} model (pretrained={pretrained})"
          f" loaded w/{model.param_counts[0]:,} params.")
    return model