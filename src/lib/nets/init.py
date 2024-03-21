import torch
import torch.nn as nn
from torch.nn import init

conv_base_class = torch.nn.modules.conv._ConvNd
linear_base_class = torch.nn.modules.linear.Linear
bn_base_class = torch.nn.modules.batchnorm._BatchNorm




def init_weights(net, init_type='normal'):
    print('Weight init method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        msg = f'initialization method {init_type} is not implemented'
        raise NotImplementedError(msg)
    return net

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, conv_base_class):
        init.normal_(m.weight.data, 0.0, 0.02)
        # init.constant_(m.bias.data, 0.0)
    elif isinstance(m, linear_base_class):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, bn_base_class):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, conv_base_class):
        init.xavier_normal_(m.weight.data, gain=1)
        # init.constant_(m.bias.data, 0.0)
    elif isinstance(m, linear_base_class):
        init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, bn_base_class):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, conv_base_class):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        # init.constant_(m.bias.data, 0.0)
    elif isinstance(m, linear_base_class):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, bn_base_class):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, conv_base_class):
        init.orthogonal_(m.weight.data, gain=1)
        # init.constant_(m.bias.data, 0.0)
    elif isinstance(m, linear_base_class):
        init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, bn_base_class):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)