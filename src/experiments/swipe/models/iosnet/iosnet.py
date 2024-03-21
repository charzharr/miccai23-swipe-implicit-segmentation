
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import numpy as np
from einops import rearrange, repeat

from typing import Dict, List, Tuple, Optional

from lib.nets.basemodel import BaseModel


class IOSNet2d(BaseModel):
    ndim = 2
    
    def __init__(self, config, encoder, decoder, displacement=0):
        super().__init__()
        
        self.config = config
        self.model_config = config.model[config.model.name]
        self.encoder = encoder 
        self.decoder = decoder
        
        # Initialize local displacements
        displacements = self._init_displacements(
            displacement, self.model_config.displacements
        )
        self.register_buffer('displacements', displacements)
        
        # Record model settings
        self.set_setting('dec_input_coords', self.model_config.dec_input_coords)
        self.set_setting('displacement', displacement)
        self.set_setting('displacements', self.displacements)
        self.set_setting('feature_flags', self.model_config.feature_flags)
        self.print_settings()
        
    def forward(self, x, p=None):
        """
        Args:
            x (tensor): BxCx...
            p (tensor):  ∈[0,1]
        """
        
        ## Image Encoder Features
        enc_d = self.encoder(x)

        ## Process Input Points (p should be Bx1xHWx2 by end)
        if p is None:  # then use full input resolution 
            rows = torch.linspace(-1, 1, steps=x.shape[-2])
            cols = torch.linspace(-1, 1, steps=x.shape[-1])
            grid_rows, grid_cols = torch.meshgrid(rows, cols, 
                                                  indexing='ij')  # HxW
            p = torch.stack([grid_rows.flatten(), grid_cols.flatten()], 
                            dim=-1)  # H*Wx2
            p = p.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) #Bx1xHWx2
            p = p.to(x.device).flip(dims=(-1,))
        else:
            assert p.ndim == 3, f'{p.shape} | should be B x #pts x 2'
            assert p.shape[0] == x.shape[0], f'{p.shape} | {x.shape}' 
            assert p.shape[-1] == self.ndim, f'{p.shape}' 
            p = p.unsqueeze(1)  # input: B x #pts x 2 -> Bx1x#ptsx2
            
            norm_pts = torch.tensor(x.shape[2:], device=x.device) - 1
            p = (p / norm_pts) * 2 - 1
            p = p.flip(dims=(-1,))  # yx -> xy
        
        ps = torch.cat([p + d for d in self.displacements], 
                       dim=-3)  # (B,1+#displacements,#pts,2)
        
        ## Get point-wise features
        
        def normalize(feat, index):
            # print('\n', feat.shape)
            # print(feat.min(), feat.max(), feat.mean())
            
            # Reduce Displacements
            feat = einops.reduce(feat, 'b c disps pts -> b c () pts', 'mean')
            if False: # weighted replacements
                num_displacements = len(self.displacements)
                weights = torch.ones(num_displacements, device=feat.device)
                weights[0] = 4
                weights = einops.rearrange(weights / weights.sum(),
                                           'd -> () () d ()')
                feat = einops.reduce(feat * weights, 
                                     'b c disps pts -> b c () pts', 'sum')
            
            def l2(feat, index):
                if index >= 1:
                    feat = feat / LA.norm(feat, dim=1).unsqueeze(1).detach()                
                    mult = 2 if index == 1 else 4
                    feat = feat * mult
                return feat
            
            def znorm(feat, index):
                mu = feat.mean(1, keepdim=True).detach()
                std = feat.std(1, keepdim=True).detach()
                feat = (feat - mu) / (std + 1e-7)
                return feat
            
            def identity(feat, index):
                return feat
            
            ## Normalize -- Feat Shape: (B, C, 1+#disp, #pts)
            # feat = l2(feat, index)
            # feat = znorm(feat, index)
            feat = identity(feat, index)
            # print(feat.min(), feat.max(), feat.mean())
            return feat
        
        features_to_cat = []
        if self.feature_flags[0]:
            features_to_cat.append(normalize(
                F.grid_sample(x, ps, align_corners=False,
                              mode=self.model_config.interp_mode), index=0)
            )  # out: (B,C_x,1+#disp,#pts)
        if self.feature_flags[1]:
            features_to_cat.append(normalize(
                F.grid_sample(self.encoder.layer0_hook.features, 
                              ps, align_corners=False,
                              mode=self.model_config.interp_mode), index=1)
            )
        if self.feature_flags[2]:
            features_to_cat.append(normalize(
                F.grid_sample(self.encoder.layer1_hook.features, 
                              ps, align_corners=False,
                              mode=self.model_config.interp_mode), index=2)
            )
        if self.feature_flags[3]:
            features_to_cat.append(normalize(
                F.grid_sample(self.encoder.layer2_hook.features, 
                              ps, align_corners=False,
                              mode=self.model_config.interp_mode), index=3)
            )
        if self.feature_flags[4]:
            features_to_cat.append(normalize(
                F.grid_sample(self.encoder.layer3_hook.features, 
                              ps, align_corners=False,
                              mode=self.model_config.interp_mode), index=4)
            )
        if self.feature_flags[5]:
            features_to_cat.append(normalize(
                F.grid_sample(self.encoder.layer4_hook.features, 
                              ps, align_corners=False,
                              mode=self.model_config.interp_mode), index=5)
            )
        if self.feature_flags[6]:
            features_to_cat.append(einops.repeat(normalize(
                F.adaptive_avg_pool2d(self.encoder.layer4_hook.features, 1),
                index=6), 'b c d pts -> b c d (repeat pts)', repeat=ps.shape[-2]
            ))
        
        # if not hasattr(self, 'iter_count'):
        #     self.iter_count = 0
        # self.iter_count += 1
        # if self.iter_count > 1000:
        #     print(
        #         feature_0.unique(),
        #         feature_1.unique(),
        #         feature_2.unique(),
        #         feature_3.unique(),
        #         feature_4.unique(),
        #         feature_5.unique(),
        #         sep='\n'
        #     )
        #     import IPython; IPython.embed(); 
            
        """
        tensor([0.0000e+00, 2.1699e-05, 3.2784e-05,  ..., 9.9990e-01, 1.0000e+00,
        1.0000e+00], device='cuda:0')
tensor([-10.9707, -10.0461,  -9.7331,  ...,   9.2933,   9.4106,   9.5070],
       device='cuda:0', grad_fn=<Unique2Backward0>)
tensor([-11.6618, -11.1855, -10.9446,  ...,  13.1863,  13.2213,  13.3806],
       device='cuda:0', grad_fn=<Unique2Backward0>)
tensor([-17.7166, -17.3402, -17.2551,  ...,  16.5528,  16.8176,  17.5932],
       device='cuda:0', grad_fn=<Unique2Backward0>)
tensor([-16.9944, -16.4540, -16.4158,  ...,  13.6565,  14.7970,  15.4042],
       device='cuda:0', grad_fn=<Unique2Backward0>)
tensor([-9.2113, -9.1289, -9.0203,  ...,  9.6767,  9.6972,  9.7858],
       device='cuda:0', grad_fn=<Unique2Backward0>)
        """
                
        ## Create Decoder Input
        # every channel corresponse to one feature.
        features = torch.cat(features_to_cat,
                              dim=1)  # (B, features, 1+#disp, #pts)
        if self.dec_input_coords:  # concatenate coordinates (?)
            # ps:       B x 1+#disp x #pts x 2
            # features: B x Cx 1+#disp x #pts
            features = torch.cat([features, ps.permute(0, -1, 1, 2)], dim=1)
        
        # Combine extra displacement features into channel dim
        shape = features.shape
        assert shape[2] == 1
        features = torch.reshape(features, # (B, feats*(1+#disp), #pts)
                                 (shape[0], shape[1] * shape[2], shape[3]))  
        
        # Implicit Shape Decoder
        logits = self.decoder(features)
        
        return {
            'out': logits, 
            'coords': ps,
            'dec_in': features,
            'enc_d': enc_d
        }
        
    def infer(self, x):
        """ Run inference that outputs a grid of logits with the same spatial
        resolution as the input.
        """
        pass

    def _init_displacements(self, displacement, displacements):
        """
        Returns: nn.Parameter of all displacements (#disp x img-ndim)
        """
        all_displacements = [[0] * self.ndim]
        
        if displacement != 0:
            if displacements:
                all_displacements.extend([[d1 * displacement, 
                                           d2 * displacement] 
                                          for d1, d2 in displacements])
            else:
                for dim in range(self.ndim):
                    for d in [-1, 1]:
                        input = [0] * self.ndim
                        input[dim] = d * displacement
                        all_displacements.append(input)
        
        return torch.tensor(all_displacements, requires_grad=False).float()



