

import torch
import torch.nn as nn


class ChannelAttn3d(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """
    def __init__(
            self, 
            channels, 
            act_factory,
            rd_ratio=1./16, 
            rd_channels=None, 
            rd_divisor=1,
            gate_layer='sigmoid', 
            mlp_bias=False):
        
        super().__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv3d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_factory.create(inplace=True)
        self.fc2 = nn.Conv3d(rd_channels, channels, 1, bias=mlp_bias)
        
        if gate_layer != 'sigmoid':
            raise NotImplementedError()
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3, 4), keepdim=True))))
        x_max = self.fc2(self.act(self.fc1(x.amax((2, 3, 4), keepdim=True))))
        return x * self.gate(x_avg + x_max)
    
    
class SpatialAttn3d(nn.Module):
    """ Original CBAM spatial attention module
    """
    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super().__init__()
        # self.conv = ConvNormAct(2, 1, kernel_size, apply_act=False)
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        
        if gate_layer != 'sigmoid':
            raise NotImplementedError()
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_attn = torch.cat([x.mean(dim=1, keepdim=True), 
                            x.amax(dim=1, keepdim=True)], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)
    
    
class CbamModule3d(nn.Module):
    def __init__(
            self, 
            channels, 
            act_factory,
            rd_ratio=1./16, 
            rd_channels=None, 
            rd_divisor=1,
            spatial_kernel_size=7, 
            gate_layer='sigmoid', 
            mlp_bias=False):
        
        super().__init__()
        self.channel = ChannelAttn3d(
            channels, act_factory, rd_ratio=rd_ratio, rd_channels=rd_channels,
            rd_divisor=rd_divisor, gate_layer=gate_layer, 
            mlp_bias=mlp_bias)
        self.spatial = SpatialAttn3d(spatial_kernel_size, gate_layer=gate_layer)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x
    

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v