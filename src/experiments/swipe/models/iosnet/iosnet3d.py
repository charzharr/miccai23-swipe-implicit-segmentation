
import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import numpy as np
from einops import rearrange, repeat

from typing import Dict, List, Tuple, Optional

from lib.nets.basemodel import BaseModel


class IOSNet3d(BaseModel):
    ndim = 3
    
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
        
        dec_global_coords = None
        if 'dec_global_coords' in self.model_config:
            dec_global_coords = self.model_config.dec_global_coords
        self.set_setting('dec_global_coords', dec_global_coords)
        
        dec_fourier_coords = nn.Identity()
        if 'dec_fourier_coords_dim' in self.model_config:
            if self.model_config.dec_fourier_coords_dim:
                from experiments.miccai23shape.utils.embeddings import SpatialEncoding
                dec_fourier_coords = SpatialEncoding(
                    self.ndim, self.model_config.dec_fourier_coords_dim,
                    require_grad=False
                )
        self.set_setting('dec_fourier_coords', dec_fourier_coords)
        
        self.print_settings()
    
    def forward(self, x, p=None, g_pnn_yx=None):
        """
        Args:
            x (tensor): BxCx...
            p (tensor):  ∈[0,1]
        """        
        ## Image Encoder Features
        enc_d = self.extract_features(x)

        ## Process Input Points (p should be Bx1xHWx2 by end)
        norm_pts = torch.tensor(x.shape[2:], device=x.device) - 1
        if p is None:  # then use full input resolution 
            deps = torch.linspace(-1, 1, steps=x.shape[-3])
            rows = torch.linspace(-1, 1, steps=x.shape[-2])
            cols = torch.linspace(-1, 1, steps=x.shape[-1])
            grid_deps, grid_rows, grid_cols = torch.meshgrid(deps, rows, cols,
                                                             indexing='ij') # DxHxW
            p = torch.stack([grid_deps.flatten(), 
                             grid_rows.flatten(),
                             grid_cols.flatten()], dim=-1)  # D*H*W x 3
            p = rearrange(p, 'dhw ndim -> 1 1 1 dhw ndim')
            p = p.repeat(x.shape[0], 1, 1, 1, 1)  # Bx1x1xDHWx3
            p = p.to(x.device).flip(dims=(-1,))   # DHW -> WHD
        else:
            assert p.ndim == 3, f'{p.shape} | should be B x #pts x 3'
            assert p.shape[0] == x.shape[0], f'{p.shape} | {x.shape}' 
            assert p.shape[-1] == self.ndim, f'{p.shape}' 
            
            p = rearrange(p, 'B P zyx -> B 1 1 P zyx')
            # p = p.unsqueeze(1)  # input: B x #pts x 3 -> Bx1x1x#ptsx3
            
            p = (p / norm_pts) * 2 - 1
            p = p.flip(dims=(-1,))  # zyx -> xyz
        
        gp_yx = torch.round(norm_pts * (p[:, 0, 0, :, :] + 1) / 2).\
                  long().flip(dims=(-1,))
        
        ps = torch.cat([p + d for d in self.displacements], 
                       dim=-3)  # (B,1,1+#displacements,#pts,3)
        
        ## Gather point-wise decoder input features
        features_l = enc_d['feature_maps']
        features = self._gather_decoder_inputs(
            x, ps, features_l, 
            input_coords=self.dec_input_coords,
            interp_mode=self.model_config.interp_mode,
            global_coords=g_pnn_yx
        )
        
        # def normalize(feat, index):
        #     # print('\n', feat.shape)
        #     # print(feat.min(), feat.max(), feat.mean())
            
        #     # Reduce Displacements
        #     feat = einops.reduce(feat, 'b c 1 disps pts -> b c () () pts', 
        #                          'mean')
        #     if False: # weighted replacements
        #         num_displacements = len(self.displacements)
        #         weights = torch.ones(num_displacements, device=feat.device)
        #         weights[0] = 4
        #         weights = einops.rearrange(weights / weights.sum(),
        #                                    'd -> () () d ()')
        #         feat = einops.reduce(feat * weights, 
        #                              'b c disps pts -> b c () pts', 'sum')
            
        #     def l2(feat, index):
        #         if index >= 1:
        #             feat = feat / LA.norm(feat, dim=1).unsqueeze(1).detach()                
        #             mult = 2 if index == 1 else 4
        #             feat = feat * mult
        #         return feat
            
        #     def znorm(feat, index):
        #         mu = feat.mean(1, keepdim=True).detach()
        #         std = feat.std(1, keepdim=True).detach()
        #         feat = (feat - mu) / (std + 1e-7)
        #         return feat
            
        #     def identity(feat, index):
        #         return feat
            
        #     ## Normalize -- Feat Shape: (B, C, 1+#disp, #pts)
        #     # feat = l2(feat, index)
        #     # feat = znorm(feat, index)
        #     feat = identity(feat, index)
        #     # print(feat.min(), feat.max(), feat.mean())
        #     return feat
                
        # features_to_cat = []
        # if self.feature_flags[0]:
        #     features_to_cat.append(normalize(
        #         F.grid_sample(x, ps, align_corners=False,
        #                       mode=self.model_config.interp_mode), index=0)
        #     )  # out: (B,C_x,1, 1+#disp,#pts)
        # if self.feature_flags[1]:
        #     features_to_cat.append(normalize(
        #         F.grid_sample(self.encoder.layer0_hook.features, 
        #                       ps, align_corners=False,
        #                       mode=self.model_config.interp_mode), index=1)
        #     )
        # if self.feature_flags[2]:
        #     features_to_cat.append(normalize(
        #         F.grid_sample(self.encoder.layer1_hook.features, 
        #                       ps, align_corners=False,
        #                       mode=self.model_config.interp_mode), index=2)
        #     )
        # if self.feature_flags[3]:
        #     features_to_cat.append(normalize(
        #         F.grid_sample(self.encoder.layer2_hook.features, 
        #                       ps, align_corners=False,
        #                       mode=self.model_config.interp_mode), index=3)
        #     )
        # if self.feature_flags[4]:
        #     features_to_cat.append(normalize(
        #         F.grid_sample(self.encoder.layer3_hook.features, 
        #                       ps, align_corners=False,
        #                       mode=self.model_config.interp_mode), index=4)
        #     )
        # if self.feature_flags[5]:
        #     features_to_cat.append(normalize(
        #         F.grid_sample(self.encoder.layer4_hook.features, 
        #                       ps, align_corners=False,
        #                       mode=self.model_config.interp_mode), index=5)
        #     )
        # if self.feature_flags[6]:
        #     features_to_cat.append(einops.repeat(
        #         normalize(F.adaptive_avg_pool3d(self.encoder.layer4_hook.features, 
        #                                         1), index=6), 
        #         'b c d h pts -> b c d h (repeat pts)', repeat=ps.shape[-2]
        #     ))
        
        # ## Create Decoder Input
        # # every channel corresponse to one feature.
        # features = torch.cat(features_to_cat,
        #                       dim=1)  # (B, features, 1+#disp, #pts)
        # if self.dec_input_coords:  # concatenate coordinates (?)
        #     # ps:       B x 1+#disp x #pts x 2
        #     # features: B x Cx 1+#disp x #pts
        #     features = torch.cat([features, ps.permute(0, -1, 1, 2, 3)], dim=1)
                
        # # Combine extra displacement features into channel dim
        # shape = features.shape
        # assert shape[2] == 1
        # features = torch.reshape(features, # (B, feats*(1+#disp), #pts)
        #                          (shape[0], np.prod(shape[1:-1]), shape[-1]))
        
        # Implicit Shape Decoder
        dec_d = self.decode_features(features)
        logits = dec_d['out']
        
        return {
            'out': logits, 
            'coords': ps,
            'dec_in': features,
            'enc_d': enc_d,
            'gp_yx': gp_yx,
        }
        
    def extract_features(self, x):
        """
        Args:
            x (input float tensor): B x C x H x W
        """
        enc_d = self.encoder(x)
        return enc_d
    
    def decode_features(self, x):
        """
        Args:
            x (decoder input float tensor): B x 
        """
        out = self.decoder(x) 
        return {'out': out}
        
    def infer(self, x):
        """ Run inference that outputs a grid of logits with the same spatial
        resolution as the input.
        """
        pass
    
    @torch.inference_mode()
    def infer_mise(
            self, 
            x, 
            resolution_stages=(4, 1),
            global_crop_info=None,
            **kwargs
            ):
        """
        Args:
            x: tensor(B x C_in x D x H x W)
            resolution_stages: sequence of ints. 
            global_crop_locations: array(B x 6), used for global dec coord input
                Each entry: [z0_tl y0_tl x0_tl z0_br y0_br x0_br]
        """
        
        assert len(resolution_stages) > 1, f'{resolution_stages}'
        
        ndim = self.ndim 
        if x.ndim == ndim + 1:
            x = x.unsqueeze(0)
        assert x.ndim == ndim + 2, f'Invalid input shape x({x.shape})'
        
        B = x.shape[0]
        channels = x.shape[1]
        size = list(x.shape[2:])
        device = x.device
        
        output_occupancies = []
        
        # Init Occupancy Grid
        def _update(occupancy_grid, coordinates, logits, size):
            B, C, num_pts = logits.shape
            
            # Init occupancy grid
            if occupancy_grid is None:
                # print('[Inf] Creating occupancy grid..')
                occupancy_grid = torch.zeros([1, C] + size, 
                                             dtype=torch.uint8, 
                                             # dtype=torch.float32, 
                                             device=logits.device)
                occupancy_grid[:, 0, :, :, :] = 1   # all background
                
            # Get class predictions and update grid
            if C == 1: 
                cid_predictions = (logits >= 0).torch.uint8       # Bx1x#pts
            else:
                cid_predictions = torch.zeros(logits.shape, device=logits.device,
                                              dtype=torch.uint8)
                cid_predictions.scatter_(1, 
                                         logits.argmax(1).unsqueeze(1),
                                         1)                          # BxCx#pts
            
            b_zs = coordinates[0, :, 0]
            b_ys = coordinates[0, :, 1]
            b_xs = coordinates[0, :, 2]
                        
            occupancy_grid[0, :, b_zs, b_ys, b_xs] = cid_predictions[0, :, :]
            
            return occupancy_grid
        
        
        for b in range(B):
            
            # Extract Image Features
            # print(f'[Inf] B{b} Sending batch element {b} into encoder.')
            enc_d = self.extract_features(x[b].unsqueeze(0))
            
            occupancy_grid = None
            
            # Perform different resolution stages
            for index, resolution_stage in enumerate(resolution_stages):
                
                # Init coordinates
                coordinates, gp_nn = self.get_gp(1, size, resolution_stage, 
                                                 device=device, ret_yx=True)
                
                # Global coordinates
                g_pnn_yx = None
                if global_crop_info is not None:
                    crop_locations = global_crop_info['locations'][b]
                    size_norm = global_crop_info['sizes'][b] - 1
                    crop_tl_nn = (crop_locations[:3] / size_norm) * 2 - 1
                    crop_br_nn = (crop_locations[3:] / size_norm) * 2 - 1
                    resolutions = 2 * resolution_stage / size_norm
                    
                    g_pnn_yx = torch.stack(
                        torch.meshgrid(
                            torch.arange(crop_tl_nn[0], crop_br_nn[0], 
                                         resolutions[0]),
                            torch.arange(crop_tl_nn[1], crop_br_nn[1], 
                                         resolutions[1]),
                            torch.arange(crop_tl_nn[2], crop_br_nn[2], 
                                         resolutions[2]),
                            indexing='ij'
                        ), dim=-1)   # D_pts, H_pts x W_pts x zyx
                    g_pnn_yx = repeat(rearrange(g_pnn_yx, 
                                                'd h w yx -> (d h w) yx'),
                                      'dhw zyx -> repeat dhw zyx', repeat=1)  
                    g_pnn_yx = g_pnn_yx.to(device)  # B x HW x yx
                    assert g_pnn_yx.shape == gp_nn.shape, \
                            f'{g_pnn_yx.shape} {gp_nn.shape}'
                
                filtered_coords = coordinates
                if index > 0:
                    # Erase occupancy grid to get coordinates for next iteration
                    kernel_size = 5 * resolution_stage
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    # kernel_size = 5 * (len(resolution_stages) - index)
                    assert kernel_size >= max(resolution_stages), f'{kernel_size}'
                    
                    binary_occupancy_grid = occupancy_grid.argmax(1, 
                                                                keepdim=True) > 0
                    occupancy_grid_erased = F.max_pool3d(
                        binary_occupancy_grid.to(torch.float16),
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2
                    )

                    possible_positions = (binary_occupancy_grid == 0) & \
                                        (occupancy_grid_erased == 1)
                    possible_positions = possible_positions[0, 0,
                                                            coordinates[0, :, 0], 
                                                            coordinates[0, :, 1],
                                                            coordinates[0, :, 2]]   # single vector

                    # Get coordinates to evaluate
                    filtered_coords = coordinates[:, possible_positions == 1]  # 1xnum_ptsx2
                    if g_pnn_yx is not None:
                        g_pnn_yx = g_pnn_yx[:, possible_positions == 1]

                # Make prediction
                # extensions = fin_res_extensions
                # extensions=None
                # if resolution_stage == 1:  # final iter
                #     extensions = fin_res_extensions
                
                if not filtered_coords.any():  # no foreground
                    break
                
                # image_enc_d = {
                #     'out': enc_d['out'][b].unsqueeze(0),
                #     'x': enc_d['x'][b].unsqueeze(0),
                #     'input_shape': [1] + list(enc_d['input_shape'])[1:],
                #     'feature_maps': [fm[b].unsqueeze(0) for fm in 
                #                      enc_d['feature_maps']]
                # }
                # print(f'[Inf] B{b}I{index} Decoding {filtered_coords.shape[1]} points '
                #       f'(batch {b}, loopidx {index})..')
                pred_d = self.infer_points(enc_d, filtered_coords, 
                                           g_pnn_yx=g_pnn_yx) 

                logits = pred_d['out']   # B x C x num_pts
                B, C, num_pts = logits.shape
                
                occupancy_grid = _update(occupancy_grid, filtered_coords, 
                                         logits, size) 
            
            output_occupancies.append(occupancy_grid)
            del pred_d; del logits; del enc_d;
            
        return {
            'out': torch.cat(output_occupancies, 0),
            # 'enc_d': enc_d
        }

    @torch.inference_mode()
    def infer_points(self, enc_d, gp_yx, g_pnn_yx=None):
        """
        Args: 
            enc_d (dict): dictionary from encoder output
                Used keys: 'out', 'patch_outs', 'out_global' (if exists)
            gp (int tensor): B x #pts x 3  int coord points in yx format
            g_pnn_yx (tensor): B x #pts x 3 normalized glob coord points [-1,1]
        """
        
        # Process gp into gp_nn and gp_nn_xy
        ndim = self.ndim
        device = enc_d['out'].device
        X_shape = enc_d['input_shape']
        B, C, D, H, W = X_shape 
        _, num_pts, _ = gp_yx.shape
        
        assert gp_yx.ndim == ndim, f'{gp_yx.shape} | should be B x #pts x 2'
        assert gp_yx.shape[0] == X_shape[0], f'{gp_yx.shape} | {X_shape}' 
        assert gp_yx.shape[-1] == ndim, f'{gp_yx.shape}' 
        
        norm_pts_yx = torch.tensor(X_shape[2:], device=device) - 1
        norm_pts_xy = norm_pts_yx.flip(dims=(-1,))
        gp_yx = gp_yx.to(device)          # B x #pts x 3
        gp_xy = gp_yx.flip(dims=(-1,))  
        gp_nn_yx = (gp_yx / norm_pts_yx) * 2 - 1
        gp_nn_xy = gp_nn_yx.flip(dims=(-1,))

        x = enc_d['x']
        ps = rearrange(gp_nn_xy, 'B P xyz -> B 1 1 P xyz')
        ps = torch.cat([ps + d for d in self.displacements], 
                       dim=-3)  # (B,1,1+#displacements,#pts,3)
        
        features_l = enc_d['feature_maps']
        features = self._gather_decoder_inputs(
            x, ps, features_l, 
            input_coords=self.dec_input_coords,
            interp_mode=self.model_config.interp_mode,
            global_coords=g_pnn_yx
        )
        
        # Implicit Shape Decoder
        dec_d = self.decode_features(features)
        logits = dec_d['out']
        
        return {
            'out': logits,
            'out_global': None,
            'enc_d': enc_d,
            'gp_xy': gp_xy,
            'gp_yx': gp_yx,
            'gp_nn_yx': gp_nn_yx,
            # 'lp': lp,
        }
        
    def get_gp(self, batch, size, resolution_stride, ret_yx=True, device='cuda'):
        """ Sample absolute and [-1,1]-normalized global points.
        Args:
            batch (int): batch size
            size (sequence): spatial dims (H, W), no channels or batch
            resolution_stride: point density downsampling factor
        Return: 
            2 long tensors (batch x np.prod(size) x yx)
            absolute_global_points, normalized_global_points [-1, 1]
        """
        ndim = self.ndim
        batch = int(batch)
        assert batch >= 1
        assert len(size) == ndim
        
        gp = torch.stack(
                torch.meshgrid(
                    torch.arange(0, size[0], resolution_stride),
                    torch.arange(0, size[1], resolution_stride),
                    torch.arange(0, size[2], resolution_stride),
                    indexing='ij'
                ), dim=-1)   # D_pts, H_pts x W_pts x zyx
        gp = repeat(rearrange(gp, 'd h w yx -> (d h w) yx'),
                    'dhw zyx -> repeat dhw zyx', repeat=batch)  # B x HW x yx
        
        norm_pts = torch.tensor(size) - 1
        gp_nn = (gp / norm_pts) * 2 - 1
        
        if device:
            gp, gp_nn = gp.to(device), gp_nn.to(device)
        
        if ret_yx:
            return gp, gp_nn
        return gp.flip(dims=(-1,)), gp_nn.flip(dims=(-1,))
    
    def _gather_decoder_inputs(
            self, 
            x,
            ps,        
            features_l,    
            input_coords=False,
            interp_mode='bilinear',
            global_coords=None      # [-1,1] normali
            ):
        """
        Args:
            x: input image tensor (B x chans x D x H x W)
            ps: tensor(B x 1 x #disp x xyz x num_pts) in xyz format [-1,1]
                Used by F.grid_sample.
            feature_maps_l (list): list of tensors(B x Ci x Di x Hi x Wi)
            global_coords: tensor(B x #pts x zyx) [-1,1] normalized global pts
        """
        assert ps.ndim == 5  # B x 1 x 1 x num_pts x 3xyz
        
        def normalize(feat, index):
            # Reduce Displacements
            feat = einops.reduce(feat, 'b c 1 disps pts -> b c () () pts', 
                                 'mean')
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
                              mode=interp_mode), index=0)
            )  # out: (B,C_x,1, 1+#disp,#pts)
        if self.feature_flags[1]:
            features_to_cat.append(normalize(
                F.grid_sample(features_l[0], 
                              ps, align_corners=False,
                              mode=interp_mode), index=1)
            )
        if self.feature_flags[2]:
            features_to_cat.append(normalize(
                F.grid_sample(features_l[1], 
                              ps, align_corners=False,
                              mode=interp_mode), index=2)
            )
        if self.feature_flags[3]:
            features_to_cat.append(normalize(
                F.grid_sample(features_l[2], 
                              ps, align_corners=False,
                              mode=interp_mode), index=3)
            )
        if self.feature_flags[4]:
            features_to_cat.append(normalize(
                F.grid_sample(features_l[3], 
                              ps, align_corners=False,
                              mode=interp_mode), index=4)
            )
        if self.feature_flags[5]:
            features_to_cat.append(normalize(
                F.grid_sample(features_l[4], 
                              ps, align_corners=False,
                              mode=interp_mode), index=5)
            )
        if self.feature_flags[6]:
            features_to_cat.append(einops.repeat(
                normalize(F.adaptive_avg_pool3d(features_l[4], 1), index=6), 
                'b c d h pts -> b c d h (repeat pts)', repeat=ps.shape[-2]
            ))
        
        ## Create Decoder Input
        # every channel corresponse to one feature.
        features = torch.cat(features_to_cat,
                              dim=1)  # (B, features, 1+#disp, #pts)
        if input_coords:  # concatenate coordinates (?)
            # ps:       B x 1+#disp x #pts x 2
            # features: B x Cx 1+#disp x #pts
            features = torch.cat([features, ps.permute(0, -1, 1, 2, 3)], dim=1)
        
        shape = features.shape
        assert shape[2] == 1
        features = torch.reshape(features, # (B, feats*(1+#disp), #pts)
                                 (shape[0], np.prod(shape[1:-1]), shape[-1]))
        
        if self.dec_global_coords:
            assert global_coords is not None
            # global_coords: B x #pts x 3, zyx format
            # features: B x C_feat x #pts
            assert global_coords.shape[0] == features.shape[0]
            assert global_coords.shape[1] == features.shape[-1]
            global_coords = global_coords.float().to(features.device)
            features = torch.cat([rearrange(global_coords, 'b p zyx -> b zyx p'),
                                  features], dim=1)
    
        return features  # B x num_feats x num_pts
    
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



class IOSNet33d(nn.Module):  # OLD.
    
    def __init__(self, config, encoder, decoder, displacement=0):
        self.config = config
        self.encoder = encoder 
        
       # Initialize local displacements
        self.displacement = displacement   # IF-Net 128vox = 0.0722
        displacements = []
        displacements.append([0, 0, 0])
        if self.displacement != 0:
            for dim in range(3):
                for d in [-1, 1]:
                    input = [0, 0, 0]
                    input[dim] = d * self.displacement
                    displacements.append(input)
        self.displacements = nn.Parameter(displacements, requires_grad=False)
        
    def forward(self, x, p):
        """
        Args:
            x (tensor): BxCx...
            p (tensor):  ∈[0,1]
        """
        
        ## Image Encoder Features
        enc_d = self.encoder(x)
        
        ## Get point-wise features
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        
        feature_0 = F.grid_sample(x, p)  # out : (B,C (of x), 1,1,sample_num)
        feature_1 = F.grid_sample(self.encoder.layer0_hook.features, p)
        feature_2 = F.grid_sample(self.encoder.layer1_hook.features, p)
        feature_3 = F.grid_sample(self.encoder.layer2_hook.features, p)
        feature_4 = F.grid_sample(self.encoder.layer3_hook.features, p)
        feature_5 = F.grid_sample(self.encoder.layer4_hook.features, p)
        
        # every channel corresponse to one feature.
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                              dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features, # (B, featues_per_sample, samples_num)
                                 (shape[0], shape[1] * shape[3], shape[4]))  
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
        
        
        ## Create Decoder Input
        features = torch.cat((feature_0, feature_1, feature_2, 
                              feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features, # (B, featues_per_sample, samples_num)
                                 (shape[0], shape[1] * shape[3], shape[4]))  
        
        # Implicit Shape Decoder
        logits = self.decoder(features)
        
        return {
            'out': logits, 
            'dec_in': features,
            'enc_d': enc_d
        }