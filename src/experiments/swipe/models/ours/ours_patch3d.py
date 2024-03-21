


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lib.nets.basemodel import BaseModel



def get_setting(name, config):
    if str(name) in config:
        return config[name] 
    return None


class PatchModel3d(BaseModel):
    ndim = 3
    
    def __init__(self, config, encoder, decoder, global_decoder=None):
        super().__init__()
        self.config = config
        self.task_config = config.task
        self.model_config = config.model[config.model.name]
        self.ndim = int(config.model.name[-2])
        
        self.encoder = encoder 
        self.decoder = decoder 
        self.global_decoder = global_decoder
        
        patch_size = 32 if 'patch_size' not in self.task_config else \
                     self.task_config.patch_size
        self.set_setting('patch_size', patch_size)      
        self.set_setting('input_size', self.encoder.img_size)
        self.set_setting('feature_flags', self.model_config.feature_flags)
        self.set_setting('local_points_rep', self.config.task.local_points_rep)
        
        # Decoder Global Coordinates
        dec_global_coords = None
        if 'dec_global_coords' in self.model_config:
            dec_global_coords = self.model_config.dec_global_coords
        self.set_setting('dec_global_coords', dec_global_coords)
        
        # Spatial Embeddings for Decoder Inputs
        from experiments.miccai23shape.utils.embeddings import SpatialEncoding
        self.set_setting('local_fourier_coords_dim', 
             get_setting('local_fourier_coords_dim', self.model_config))
        self.local_fourier_coords = None 
        if self.local_fourier_coords_dim:
            self.local_fourier_coords = SpatialEncoding(
                self.ndim, 
                self.local_fourier_coords_dim,
                require_grad=self.model_config.learnable_fourier_coords
            )
        
        self.set_setting('global_fourier_coords_dim', 
             get_setting('global_fourier_coords_dim', self.model_config))
        self.global_fourier_coords = None 
        if self.global_fourier_coords_dim:
            self.global_fourier_coords = SpatialEncoding(
                self.ndim, 
                self.global_fourier_coords_dim,
                require_grad=self.model_config.learnable_fourier_coords
            )
        
        self.print_settings()
        
        
    def forward(self, x, p=None, g_pnn_yx=None):
        ## Get Patch Embeddings
        enc_d = self.extract_features(x)
        global_feats = enc_d['global_out']
        # B, P, D = enc_d['out'].shape  # batch, num_patches, embedding_dim
        
        ## Get Decoder Predictions
        P_d = self._process_points(x, p)
        orig_gp = P_d['orig_gp']
        gp = P_d['gp']
        lp = P_d['lp']
        patch_indices = P_d['patch_indices']
        num_original_points = P_d['num_original_points']
        
        gp_feats = rearrange(gp, 'b 1 np xy -> b xy np')
        lp_feats = rearrange(lp, 'b 1 np xy -> b xy np')
        patch_indices = rearrange(patch_indices, 'b 1 np yx -> b yx np')
        num_pts = lp_feats.shape[-1]
        num_h_patches = int(self.input_size[0] / self.patch_size)
        
        # import IPython; IPython.embed(); 
        
        dec_in, glob_dec_in = self._gather_decoder_inputs(
            gp_feats, lp_feats, global_feats, enc_d['patch_outs'], 
            patch_indices, num_h_patches,
            num_global_samples=num_original_points,
            global_coords=g_pnn_yx
        )
        dec_d = self.decode_features(dec_in, glob_x=glob_dec_in)
        
        return {
            'out': dec_d['out'],
            'out_global': dec_d['global_out'],
            'enc_d': enc_d,
            'gp_yx': P_d['gp_yx'],
            'gp': gp,
            'lp': lp
        }
        
    def extract_features(self, x):
        """
        Args:
            x (input float tensor): B x C x H x W
        """
        enc_d = self.encoder(x)
        return enc_d
    
    def decode_features(self, x, glob_x=None):
        """
        Args:
            x (decoder input float tensor): B x 
        """
        out = self.decoder(x) 
        global_out = self.global_decoder(glob_x) if glob_x is not None else None
    
        return {
            'out': out,
            'global_out': global_out
        }
    
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
                
                if filtered_coords.shape[1] > 300000:
                    print(f'[Inf] B{b}I{index} Decoding {filtered_coords.shape[1]} points '
                    f'(batch {b}, loopidx {index})..')
                    pred_d = self.infer_points(enc_d, filtered_coords[:, :300000], 
                                                g_pnn_yx=g_pnn_yx[:, :300000],
                        # extensions=[[0, 1, 0], [0, -1, 0], [1, 0, 0],[-1, 0, 0],]
                                            ) 
                    logits = pred_d['out']  # B x C x num_pts
                    B, C, num_pts = logits.shape
                    occupancy_grid = _update(occupancy_grid, 
                                             filtered_coords[:, :300000], 
                                             logits, size) 
                    
                    pred_d = self.infer_points(enc_d, filtered_coords[:, 300000:], 
                                               g_pnn_yx=g_pnn_yx[:, 300000:],
                        # extensions=[[0, 1, 0], [0, -1, 0], [1, 0, 0],[-1, 0, 0],]
                                            ) 
                    logits = pred_d['out']   # B x C x num_pts
                    B, C, num_pts = logits.shape
                    occupancy_grid = _update(occupancy_grid, 
                                                filtered_coords[:, 300000:], 
                                                logits, size) 
                        
                else:
                    # print(f'[Inf] B{b}I{index} Decoding {filtered_coords.shape[1]} points '
                    #     f'(batch {b}, loopidx {index})..')
                    pred_d = self.infer_points(enc_d, filtered_coords, 
                                            g_pnn_yx=g_pnn_yx,
                        # extensions=[[0, 1, 0], [0, -1, 0], [1, 0, 0],[-1, 0, 0],]
                                            ) 
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
    def infer_points(self, enc_d, gp_yx, g_pnn_yx=None, extensions=[]):
        """
        Args: 
            enc_d (dict): dictionary from encoder output
                Used keys: 'out', 'patch_outs', 'out_global' (if exists)
            gp (int tensor): B x #pts x 3  int coord points in yx format
            g_pnn_yx (tensor): B x #pts x 3 normalized glob coord points [-1,1]
        """
        
        # Process gp into gp_nn and gp_nn_xy
        ndim = self.ndim
        global_embedding = enc_d['global_out']  # B x embed_dim
        enc_out = enc_d['out']
        device = enc_out.device if isinstance(enc_out, torch.Tensor) \
                 else enc_out[-1].device
        X_shape = enc_d['input_shape']
        B, C, D, H, W = X_shape 
        _, num_patches, dim = enc_out.shape if isinstance(enc_out, torch.Tensor) \
                              else enc_out[-1].shape  # B, num_patches, embed_dim
        _, num_pts, _ = gp_yx.shape
        num_patches_per_dim = round(num_patches ** (1/ndim))
        patch_size = H // num_patches_per_dim
        
        
        assert gp_yx.ndim == ndim, f'{gp_yx.shape} | should be B x #pts x 2'
        assert gp_yx.shape[0] == X_shape[0], f'{gp_yx.shape} | {X_shape}' 
        assert gp_yx.shape[-1] == ndim, f'{gp_yx.shape}' 
        
        norm_pts_yx = torch.tensor(X_shape[2:], device=device) - 1
        norm_pts_xy = norm_pts_yx.flip(dims=(-1,))
        gp_yx = gp_yx.to(device)          # B x #pts x 3
        gp_xy = gp_yx.flip(dims=(-1,))  
        gp_nn_yx = (gp_yx / norm_pts_yx) * 2 - 1
        gp_nn_xy = gp_nn_yx.flip(dims=(-1,))

        # Get patch indices
        # rel_patch_size = 2 * patch_size / (H - 1)
        # orig_coords_xy = torch.round(norm_pts_xy * (gp_nn_xy + 1) / 2)
        patch_indices_yx = torch.floor(gp_yx / patch_size).contiguous().long()
        
        # Iterate through patch extensions
        if not extensions:
            extensions = [(0, 0, 0)]   # yx
        extensions = torch.tensor(extensions, device=device)
                
        final_logits, logits_mask = None, None
        def _update(final_logits, logits_mask, dec_out, valid_mask,
                    num_extensions, index):
            B, C, Pts = dec_out.shape
            if final_logits is None:
                final_logits = torch.zeros((B, C, num_pts, num_extensions),
                                           device=device, dtype=torch.float32)
                logits_mask = torch.zeros((B, num_pts, num_extensions),
                                          device=device, dtype=torch.bool)
            
            # Update Values 
                # dec_out: B x classes x num_pts
                # valid_mask: B x num_pts
            final_logits[:, :, :, index] = dec_out
            logits_mask[:, :, index] = valid_mask
            
            return final_logits, logits_mask
        
        for i, ext in enumerate(extensions):
            ext_PI_yx = patch_indices_yx + ext   # B x 1 x #pts x 3
            
            # Update validity mask
            valid_mask = (ext_PI_yx >= 0) & (ext_PI_yx < num_patches_per_dim)
            valid_mask = valid_mask[..., 0] & valid_mask[..., 1]  # B x npt
            
            # Clip global indices and get local indices
            final_PI_yx = ext_PI_yx.clip(0, num_patches_per_dim - 1)  
            
            # Get lp_nn_xy from gp_nn_xy and patch centers
            patch_centers_xy = (final_PI_yx * patch_size + \
                                0.5 * patch_size).flip(dims=(-1,))       # B x npt x 3
            patch_centers_nn_xy = (patch_centers_xy/norm_pts_xy)*2 - 1   # B x npt x 3
            lp_nn_xy = gp_nn_xy - patch_centers_nn_xy                    # xy
            # if i == 0:
            #    lp_xy = torch.round(norm_pts_xy * (lp_nn_xy + 1) / 2)
            
            in_global_embedding = global_embedding
            # if i > 0:  # Experimental dropout
            #     in_global_embedding = F.dropout(global_embedding, p=0.0)
            
            # Gather decoder input features            
            dec_in, glob_dec_in = self._gather_decoder_inputs(
                gp_feats=rearrange(gp_nn_xy, 'b np xy -> b xy np'),
                lp_feats=rearrange(lp_nn_xy, 'b np xy -> b xy np'),
                global_embedding=in_global_embedding,
                patch_embeddings_l=enc_d['patch_outs'],
                patch_indices=rearrange(final_PI_yx, 'b np yx -> b yx np'),
                num_patches_per_dim=num_patches_per_dim, 
                global_coords=g_pnn_yx
            )  # B x feat_dim x num_pts
            
            # Predict via Decoder 
            dec_d = self.decode_features(dec_in, glob_x=None)
            glob_dec_out = dec_d['global_out']  # B x classes x num_pts or None
            dec_out = dec_d['out']            # B x classes x num_pts
            
            # Update logits and validity masks
            final_logits, logits_mask = _update(final_logits, logits_mask, 
                                                dec_out, valid_mask,
                                                len(extensions), i)
        
        # Compute logits average and variance
        
        logits_mask_sum = logits_mask.sum(dim=-1)  # B x Npt x Next -> B x Npt
        mean_logits = torch.divide((logits_mask * final_logits).sum(dim=-1),
                                   logits_mask_sum)  # B x C x Npts
        
        var_logits = ((final_logits - mean_logits.unsqueeze(-1)) ** 2) * logits_mask 
        var_logits = var_logits.sum(dim=-1) / logits_mask_sum
        
        return {
            'out': mean_logits,
            'out_global': dec_d['global_out'],
            'enc_d': enc_d,
            'gp_xy': gp_xy,
            'gp_yx': gp_yx,
            'gp_nn_yx': gp_nn_yx,
            # 'lp': lp,
            'logits_all': final_logits,
            'logits_mask': logits_mask,
            'logits_mean': mean_logits,
            'logits_var': var_logits 
        }
        
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

    def _process_points(self, x, p):
        shape = x.shape
        
        ## Global points gp should be Bx1xHWx2 by end
        norm_pts = torch.tensor(x.shape[2:], device=x.device) - 1
        if p is None:
            deps = torch.linspace(-1, 1, steps=shape[-3])
            rows = torch.linspace(-1, 1, steps=shape[-2])
            cols = torch.linspace(-1, 1, steps=shape[-1])
            grid_deps, grid_rows, grid_cols = torch.meshgrid(deps, rows, cols, 
                                                             indexing='ij')  # HxW
            gp = torch.stack([grid_deps.flatten(), 
                              grid_rows.flatten(), 
                              grid_cols.flatten()], dim=-1)  # H*Wx3
            gp = gp.unsqueeze(0).unsqueeze(0).repeat(shape[0], 1, 1, 1) #Bx1xDHWx2
            gp = gp.to(x.device).flip(dims=(-1,))
        else:
            assert p.ndim == 3, f'{p.shape} | should be B x #pts x 3'
            assert p.shape[0] == x.shape[0], f'{p.shape} | {x.shape}' 
            assert p.shape[-1] == self.ndim, f'{p.shape}' 
            p = p.unsqueeze(1)  # input: B x #pts x 3 -> Bx1x#ptsx3
            
            gp = (p / norm_pts) * 2 - 1
            gp = gp.flip(dims=(-1,))  # zyx -> xyz
        
        orig_gp = gp
        
        ## Get patch indices -> B x rows x cols x #pts  
        B, _, num_pts, ndim = gp.shape
        assert shape[-1] == shape[-2] == shape[-3], \
                f'TODO: make for arbitrary patch sizes'
        rel_patch_size = 2 * self.patch_size / (shape[-1] - 1)
        orig_coords = torch.round(norm_pts * (gp + 1) / 2)  # format: xy
        
        patch_indices = torch.floor(orig_coords / self.patch_size)
        # patch_indices = rearrange(patch_indices.flip(dims=(-1,)), 
        #                           'b 1 p yx -> b yx p').contiguous().long()
        patch_indices = patch_indices.flip(dims=(-1,)).contiguous().long()
        
        ## Get local patch points
        if 'norm' in self.local_points_rep:
            lp = (orig_coords % self.patch_size) / (self.patch_size - 1)
            lp = lp * 2 - 1   # xy ∈ [-1, 1]
            
            # lp = (gp % rel_patch_size) / rel_patch_size   # lp ∈ [0, 1]
            # denom = (self.patch_size - 1) / (shape[-1] - 1)
            # lp = ((gp + 1) % rel_patch_size) / denom - 1     # lp ∈ [-1, 1]
        elif 'center' in self.local_points_rep:
            
            # Get local points by center 
            max_patches = int(self.input_size[0] / self.patch_size)
            num_extensions = 0 if p is None else self.task_config.num_extensions
            
            if num_extensions > 0:
                choices = torch.tensor(
                    [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0],
                     [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                     [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, -1, 0], [1, 0, 0],
                     [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                     [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, -1, 0], [-1, 0, 0],
                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], 
                    dtype=torch.int32, device=x.device
                )
            
            gps, lps, pis = [], [], []
            for i in range(num_extensions + 1):
                new_patch_indices = patch_indices.clone()  # yx
                if i > 0:  # perturb 
                    choice = torch.randint(low=0, high=choices.shape[0], 
                                           size=(B, 1, num_pts))
                    perturb = choices[choice]

                    new_patch_indices = new_patch_indices + perturb
                    new_patch_indices[new_patch_indices < 0] *= -1
                    new_patch_indices[new_patch_indices >= max_patches] -= 1
                
                patch_centers = (new_patch_indices * self.patch_size + \
                                 0.5 * self.patch_size).flip(dims=(-1,))  
                rel_patch_centers = (patch_centers / norm_pts) * 2 - 1  # xy
                
                lp = gp - rel_patch_centers  # xy
                
                gps.append(gp)
                lps.append(lp)
                pis.append(new_patch_indices)
                            
            gp = torch.cat(gps, -2)  # Shape: Bx1xNPx2
            lp = torch.cat(lps, -2)
            patch_indices = torch.cat(pis, -2)
        else:
            raise ValueError(f'{self.local_points_rep} not valid.') 
        
        orig_coords_xy = torch.round(norm_pts * (gp + 1) / 2).long()
        gp_xy = orig_coords_xy[:, 0]
        gp_yx = orig_coords_xy[:, 0].flip(dims=(-1,))
        
        assert orig_gp.is_contiguous()
        assert gp.is_contiguous()
        assert lp.is_contiguous()
        assert patch_indices.is_contiguous()
        
        return {
            'num_original_points': orig_gp.shape[-2],
            'orig_gp': orig_gp,
            'gp': gp,
            'lp': lp,
            'patch_indices': patch_indices,
            'gp_xy': gp_xy,
            'gp_yx': gp_yx
        }
    
    def _gather_decoder_inputs(
            self, 
            gp_feats,            
            lp_feats,
            global_embedding,   
            patch_embeddings_l,  
            patch_indices,       # tensor(B x num_pts x ndim)
            num_patches_per_dim, # number of patches along a dim, not patch size
            num_global_samples=None,
            global_coords=None   # yx
            ):
        """
        Args:
            gp_feats: tensor(B x xy x num_pts) in xy format
            lp_feats: tensor(B x xy x num_pts) in xy format
            global_embedding: tensor (B x emb)
            patch_embeddings_l (list): list of tensors(B x num_patches x emb)
            patch_indices:  tensor(B x yx x num_pts) in yx format
            num_patches_per_dim (int): number of patches along either dim
        """
        
        if patch_indices.ndim == 4:
            patch_indices = rearrange(patch_indices, 'b 1 np yx -> b yx np')
        B, ndim, num_pts = lp_feats.shape
        num_h_patches = num_patches_per_dim
        
        if isinstance(patch_embeddings_l[-1], torch.Tensor):
            D = patch_embeddings_l[-1].shape[-1]
        else:
            D = patch_embeddings_l[-1][-1].shape[-1]
        
        if isinstance(global_embedding, torch.Tensor):
            global_feats = global_embedding.unsqueeze(-1).repeat(1, 1, num_pts)
        else:
            global_feats = [emb.unsqueeze(-1).repeat(1, 1, num_pts) 
                            for emb in global_embedding]
                
        
        def _get_dec_features(feats, patch_indices,
                              num_h_patches, B, num_pts, D):
            
            patch_feats = rearrange(feats, 'b (d h w) dim -> b d h w dim',
                                    d=num_h_patches, h=num_h_patches)
            local_feats = torch.zeros((B, num_pts, D), device=feats.device)
            for b in range(B):
                local_feats[b] = patch_feats[b, patch_indices[b, 0, :], 
                                                patch_indices[b, 1, :], 
                                                patch_indices[b, 2, :], :]
            local_feats = rearrange(local_feats, 'b pts emb -> b emb pts')
            return local_feats
        
        dec_in = []
        loc_feats = []
        if self.feature_flags[0]: 
            dec_in.append(gp_feats)
        if self.feature_flags[2]: 
            dec_in.append(lp_feats)
        if self.feature_flags[3]: 
            if isinstance(patch_embeddings_l[-1], torch.Tensor):
                loc_feats.append(_get_dec_features(patch_embeddings_l[-1], 
                                                   patch_indices,
                                                   num_h_patches, B, num_pts, D))
            else:
                for loc_emb in patch_embeddings_l[-1]:
                    loc_feats.append(_get_dec_features(loc_emb, 
                                                       patch_indices,
                                                       num_h_patches, B, num_pts, D))
                
        # Gather local decoder inputs 
        if self.local_fourier_coords is not None:
            lp = rearrange(lp_feats, 'b xy p -> b p xy')
            lp_fourier = rearrange(self.local_fourier_coords(lp),
                                   'b p fourier -> b fourier p')
            dec_in.append(lp_fourier)
        if self.global_fourier_coords is not None:
            gp = rearrange(gp_feats, 'b xy p -> b p xy')
            gp_fourier = rearrange(self.global_fourier_coords(gp),
                                   'b p fourier -> b fourier p')
            dec_in.append(gp_fourier)
        if self.dec_global_coords:
            assert global_coords is not None
            # global_coords: B x #pts x 3, zyx format
            # features: B x C_feat x #pts
            global_coords = global_coords.float().to(dec_in[-1].device)
            B, _, num_pts = dec_in[-1].shape
            assert global_coords.shape[0] == B
            global_coords = rearrange(global_coords, 'b p zyx -> b zyx p')
            
            if num_pts != global_coords.shape[-1]:
                global_coords = repeat(global_coords, 'b zyx p -> b zyx (r p)',
                                       r=num_pts // global_coords.shape[-1])
            assert global_coords.shape[-1] == num_pts
            dec_in.append(global_coords)
        
        dec_in = torch.cat(dec_in, dim=1)
        
        # Gather local features 
        if isinstance(patch_embeddings_l[-1], torch.Tensor):
            cat = [dec_in, loc_feats[0]]
            if self.feature_flags[1]:
                cat.append(global_feats)
            dec_in = torch.cat(cat, 1)
        else:
            new_dec_in = []
            for ci, loc_feat in enumerate(loc_feats):
                cat = [dec_in, loc_feat]
                if self.feature_flags[1]:
                    cat.append(global_feats[ci])
                new_dec_in.append(torch.cat(cat, 1))
            dec_in = new_dec_in

        
        # Gather global decoder inputs
        if isinstance(global_embedding, torch.Tensor):
            if num_global_samples:
                glob_dec_in = torch.cat([gp_feats[:, :, :num_global_samples], 
                                        global_feats[:, :, :num_global_samples]], 
                                        dim=1)
            else:
                glob_dec_in = torch.cat([gp_feats, global_feats], dim=1)
        else:
            if num_global_samples:
                glob_dec_in = [torch.cat([gp_feats[:, :, :num_global_samples], 
                                          glob_emb[:, :, :num_global_samples]], 
                                          dim=1) 
                               for glob_emb in global_feats]
            else:
                glob_dec_in = [torch.cat([gp_feats, glob_emb], dim=1) 
                               for glob_emb in global_feats]
                
        return dec_in, glob_dec_in
        
    