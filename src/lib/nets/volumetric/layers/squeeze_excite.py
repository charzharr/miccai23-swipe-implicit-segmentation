
import torch
import torch.nn as nn


class SEModule3d(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, 
            channels, 
            act_factory,
            rd_ratio=1. / 16, 
            rd_channels=None, 
            rd_divisor=8, 
            add_maxpool=False,
            bias=True, 
            norm_layer=None, 
            gate_layer='sigmoid',
            ):
        
        super(SEModule3d, self).__init__()
        
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, 
                                         rd_divisor, 
                                         round_limit=0.)
        
        self.fc1 = nn.Conv3d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act_factory.create(inplace=True)
        self.fc2 = nn.Conv3d(rd_channels, channels, kernel_size=1, bias=bias)
        
        if gate_layer != 'sigmoid':
            raise NotImplementedError()
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3, 4), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
    
    
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
