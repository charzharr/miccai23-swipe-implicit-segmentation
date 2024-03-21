"""
Implicit shape decoders
"""

import torch 
import torch.nn as nn

from lib.nets.basemodel import BaseModel


class IOSDecoder(BaseModel):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_dims=[1024, 1024, 1024],
            ):
        super().__init__()
        
        self.set_setting('in_channels', in_channels)
        self.set_setting('out_channels', out_channels)
        self.set_setting('hidden_dims', hidden_dims)
        
        layers = []
        for i in range(len(hidden_dims) + 1):
            in_dims = in_channels if i == 0 else hidden_dims[i-1]
            out_dims = out_channels if i == len(hidden_dims) else hidden_dims[i]
            layers.append(nn.Conv1d(in_dims, out_dims, 1))
            
            if i < len(hidden_dims):
                layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*layers)
        
        self.print_settings()
        
    def forward(self, x):
        x = self.decoder(x)
        return x


class MLPDecoder(BaseModel):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_dims=[1024, 1024, 1024],
            ):
        super().__init__()
        
        self.set_setting('in_channels', in_channels)
        self.set_setting('out_channels', out_channels)
        self.set_setting('hidden_dims', hidden_dims)
        
        layers = []
        for i in range(len(hidden_dims) + 1):
            in_dims = in_channels if i == 0 else hidden_dims[i-1]
            out_dims = out_channels if i == len(hidden_dims) else hidden_dims[i]
            layers.append(nn.Linear(in_dims, out_dims))
            
            if i < len(hidden_dims):
                layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*layers)
        
        self.print_settings()
        
    def forward(self, x):
        x = self.decoder(x)
        return x