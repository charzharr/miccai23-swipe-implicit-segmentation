"""
Implicit shape decoders
"""

import torch 
import torch.nn as nn

from lib.nets.basemodel import BaseModel



class DeepSDFDecoder(BaseModel):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_dims=[512, 512, 512, 512, 512, 512, 512, 512],
            residual_layers=[4],
            dropout=0.2,
            norm='wn'   # 'bn' 'ln' 'none' 'wn'
            ):

        super().__init__()
        
        self.set_setting('in_channels', in_channels)
        self.set_setting('out_channels', out_channels)
        self.set_setting('hidden_dims', hidden_dims)
        self.set_setting('residual_layers', residual_layers)
        self.set_setting('norm', norm)
        self.set_setting('dropout', dropout)
        
        all_layers = []
        for i in range(len(hidden_dims) + 1):
            layers = []
            in_dims = in_channels if i == 0 else hidden_dims[i-1]
            if residual_layers and i in residual_layers:
                in_dims += in_channels
            out_dims = out_channels if i == len(hidden_dims) else hidden_dims[i]
            
            if norm == 'wn':
                layers.append(nn.utils.weight_norm(nn.Conv1d(in_dims, out_dims, 1)))
            else:
                layers.append(nn.Conv1d(in_dims, out_dims, 1))
            
            if i < len(hidden_dims):
                if norm == 'bn':
                    layers.append(nn.BatchNorm1d(out_dims))
                elif norm == 'ln':
                    layers.append(nn.LayerNorm(out_dims))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0 and i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(p=dropout))
            all_layers.append(nn.Sequential(*layers))
        # self.decoder = nn.Sequential(*layers)
        self.decoder = nn.ModuleList(all_layers)
        self.print_settings()
        self.apply(self.init_weight)
        
    def forward(self, x):
        # x = self.decoder(x)
        # return x
        
        inp = x
        for i, layer in enumerate(self.decoder):
            if i in self.residual_layers:
                x = torch.cat([x, inp], 1)
            x = layer(x)
            
        return x
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
            
class MCDeepSDFDecoder(BaseModel):
    """ Multi-class SDF-like implicit shape decoder """
    
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_dims=[512, 512, 512, 512, 512, 512, 512, 512],
            residual_layers=[4],
            dropout=0.2,
            norm='wn',   # 'bn' 'ln' 'none' 'wn'
            d=128
            ):

        super().__init__()
        
        self.set_setting('in_channels', in_channels)
        self.set_setting('out_channels', out_channels)
        self.set_setting('hidden_dims', hidden_dims)
        self.set_setting('d', d)
        self.set_setting('residual_layers', residual_layers)
        self.set_setting('norm', norm)
        self.set_setting('dropout', dropout)
        
        assert out_channels > 1, f'{out_channels}'
        decoders = []
        for ci in range(out_channels):
            decoders.append(DeepSDFDecoder(
                in_channels,
                1,
                hidden_dims=hidden_dims,
                residual_layers=residual_layers,
                dropout=dropout,
                norm=norm
            ))
        
        self.decoders = nn.ModuleList(decoders)
        self.print_settings()
        self.apply(self.init_weight)
        
    def forward(self, x):
        # x = self.decoder(x)
        # return x
                
        class_scores = [] 
        for ci, decoder in enumerate(self.decoders):
            class_scores.append(decoder(x[ci]))
        
        x = torch.cat(class_scores, 1)
                    
        return x
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        
        

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
            layers.append(nn.Conv1d(in_dims, out_dims, 1))
            
            if i < len(hidden_dims):
                layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*layers)
        
        self.print_settings()
        
    def forward(self, x):
        x = self.decoder(x)
        return x

