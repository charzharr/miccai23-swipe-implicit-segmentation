"""
Modified from: 
    https://github.com/hzhupku/IFA/blob/main/pyseg/models/ifa_utils.py
"""

import torch 
import torch.nn as nn 
import numpy as np 
from einops import rearrange, repeat


class SpatialEncoding(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 sigma=6,
                 cat_input=True,
                 require_grad=False
                 ):

        super().__init__()
        
        self.out_dim = out_dim 
        if self.out_dim:
            assert out_dim % (2*in_dim) == 0, "dimension must be divisible"
            
            n = out_dim // 2 // in_dim
            m = 2 ** np.linspace(0, sigma, n)
            m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
            m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], 
                            axis=0)
            
            self.emb = torch.FloatTensor(m)
            if require_grad:
                self.emb = nn.Parameter(self.emb, requires_grad=True)    
            self.in_dim = in_dim
            self.sigma = sigma
            self.cat_input = cat_input
            self.require_grad = require_grad

    def forward(self, p, cat_input=None):
        """
        Args:
            p: tensor(..., ndim)
        """
        if not self.out_dim:
            return cat_input
        
        if not self.require_grad:
            self.emb = self.emb.to(p.device)
        y = torch.matmul(p, self.emb.T)
        if cat_input is not None:
            return torch.cat([cat_input, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
    
    # def forward(self, x, cat_input=True):
       
    #     if not self.require_grad:
    #         self.emb = self.emb.to(x.device)
    #     y = torch.matmul(x, self.emb.T)
    #     if self.cat_input:
    #         return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
    #     else:
    #         return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        
        
if __name__ == '__main__':
    se2d = SpatialEncoding(2, 24)
        