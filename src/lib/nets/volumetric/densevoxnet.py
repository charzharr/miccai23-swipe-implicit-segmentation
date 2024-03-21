""" Module nets/volumetric/densevoxnet.py

DenseVoxNet architecture. 
- nc=16, k=12, N=12, params=6.8M.
"""

import torch
import torch.nn as nn

from ..basemodel import BaseModel


class _Dense_Blk(nn.Module):
    def __init__(self, in_channels, k=12, dropout=False):
        super(_Dense_Blk, self).__init__()
        layers = [
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, k, kernel_size=3, padding=1, bias=False),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        residual = x
        x_conv = self.model(x)
        out = torch.cat([residual, x_conv], 1)
        return out

    
class _Dense_Stack(nn.Module):
    def __init__(self, in_channels, k=12, N=12, dropout=False):
        super(_Dense_Stack, self).__init__()
        layers = []
        self.curc = in_channels
        for i in range(N):
            layers += [_Dense_Blk(self.curc, k)]
            #layers.append(_Dense_Blk(curc,k))
            self.curc += k
        self.encode = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encode(x), self.curc


class DenseVoxNet(BaseModel):
    def __init__(self, in_channels, num_classes, nc=16, k=12, N=12, 
                 deep_sup=False):
        """
        Args:
            in_channels: # input dimensions
            num_classes: # output dimensions
            nc: initial conv channels
            k: growth rate
            N: multiplier
        """

        self.deep_sup = deep_sup
        self.deep_sup_resolutions = [1]

        super(DenseVoxNet, self).__init__()
        self.scale1 = nn.Conv3d(in_channels, nc, kernel_size=3, stride=2, 
                                padding=1)
        self.scale2 = _Dense_Stack(nc, N=N)
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(nc+N*k),
            nn.ReLU(),
            nn.Conv3d(nc+N*k, nc+N*k, kernel_size=1),
        )
        self.mp2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.scale3 = _Dense_Stack(nc+N*k, N=12)

        if self.deep_sup:
            self.dec2 = nn.Sequential(
                nn.BatchNorm3d(nc+N*k),  # added by CZ, 12/6/21
                nn.ReLU(inplace=True),   # added by CZ, 12/6/21
                nn.Conv3d(nc+N*k, 64, kernel_size=3, 
                          padding=1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(64, 64, kernel_size=4, 
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, num_classes, kernel_size=1),
            )
        self.dec3 = nn.Sequential(
            nn.BatchNorm3d(nc+N*k*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(nc+N*k*2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, 
                               padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_classes, kernel_size=1),
        )
    
        tot_params, tot_tparams = self.param_counts
        print(f'💠 DVN-3D model initiated with n_classes={num_classes}, '
              f'(deep_sup={deep_sup})\n'
              f'   n_input={in_channels}, nc={nc}, k={k}, N={N}\n'
              f'   params={tot_params:,}, trainable_params={tot_tparams:,}.')
    
    def forward(self, x):
        s1 = self.scale1(x)
        s2, c2 = self.scale2(s1)
        s2_trans = self.trans2(s2)  # unact

        s2_pool = self.mp2(s2_trans)
        s3, c3 = self.scale3(s2_pool)  # unact
        
        # Upsample s2_trans and s3 for logits output
        up3 = self.dec3(s3)

        if self.deep_sup:
            up2 = self.dec2(s2_trans)
            return {'out': up3, '2x': up2}  # both are input shape
        return {'out': up3}
    
    
def get_model(in_channels, num_classes, nc=16, k=12, N=12, deep_sup=False):
    model = DenseVoxNet(in_channels, num_classes, nc=nc, k=k, N=N, 
                        deep_sup=deep_sup)
    return model

    