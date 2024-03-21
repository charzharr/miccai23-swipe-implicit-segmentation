

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, repeat

from lib.nets.basemodel import BaseModel



# ========================================================================== #
# * ### * ### * ### *            Basic Modules           * ### * ### * ### * #
# ========================================================================== #


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, 
                 stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    
class Downsample2x(nn.Module):
    def __init__(self, in_channels, out_channels, mode='max_pool'):
        """
        Args:
            mode: max_pool, avg_pool, conv, interp
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        
        if 'conv' in self.mode:
            self.down_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        elif 'interp' in self.mode:
            self.down_layer = nn.Upsample(scale_factor=0.5, model='bilinear',
                                          align_corners=False)
        
    def forward(self, x):
        shape = x.shape
        size = shape[2:]
        out_size = tuple(np.divide(size, 2).astype('int32').tolist())
        if 'avg' in self.mode:
            x = F.adaptive_avg_pool2d(x, out_size)
        elif 'max' in self.mode:
            x = F.max_pool2d(x, 2, stride=2)
        else:
            x = self.down_layer(x)
            
        return x
            
    


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class RFB_lite(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, out_channels):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, out_channels, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x
    

class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 attention_dropout=0.1, 
                 projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', 
                                          h = self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))
    
    @staticmethod 
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) 
                                for i in range(dim)] for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return rearrange(pe, '... -> 1 ...')
    
    
# ========================================================================== #
# * ### * ### * ### *    Semantic Accumulator Modules    * ### * ### * ### * #
# ========================================================================== #
    

class ConvAccumulator(BaseModel):
    
    def __init__(self, patch_emb_dim, hidden_dim, use_rfb=True):
        self.set_setting('patch_emb_dim', patch_emb_dim)
        self.set_setting('hidden_dim', hidden_dim)
        self.set_setting('use_rfb', use_rfb)
        super().__init__()
        
        self.model = nn.Sequential(
                nn.Conv2d(patch_emb_dim * 4, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True),
                RFB(
                    hidden_dim, 
                    hidden_dim) if self.use_rfb else nn.Identity(),
                nn.Conv2d(hidden_dim, patch_emb_dim, 1)
            )
        
        self.print_settings()
        
    def forward(self, features_list):
        x = torch.cat(features_list, 1)
        x = self.model(x) 
        return x


class SEAccumulator(BaseModel):
    
    def __init__(self, patch_emb_dim, hidden_dim, use_rfb=True):
        self.set_setting('patch_emb_dim', patch_emb_dim)
        self.set_setting('hidden_dim', hidden_dim)
        self.set_setting('use_rfb', use_rfb)
        super().__init__()
        
        self.model = nn.Sequential(
                nn.Conv2d(patch_emb_dim * 4, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True),
                RFB(
                    hidden_dim, 
                    hidden_dim) if self.use_rfb else nn.Identity(),
                # nn.Conv2d(hidden_dim, patch_emb_dim, 1)
            )
        
        self.reduction_conv = nn.Conv2d(hidden_dim, hidden_dim // 4, 1)
        self.act = nn.ReLU(True)
        self.attention_score_conv = nn.Conv2d(hidden_dim // 4, hidden_dim, 1)
        
        self.out = nn.Conv2d(hidden_dim, patch_emb_dim, 1)
        
        self.print_settings()
        
    def forward(self, features_list):
        x = torch.cat(features_list, 1)
        x = self.model(x) 
        
        se_x = self.act(self.reduction_conv(x))
        se_x = x * self.attention_score_conv(se_x).sigmoid()
        
        x = self.act(x + se_x)
        x = self.out(x)
        
        return x



class ReweighAccumulator(BaseModel):
    
    def __init__(self, patch_emb_dim, hidden_dim, use_rfb=True):
        self.set_setting('patch_emb_dim', patch_emb_dim)
        self.set_setting('hidden_dim', hidden_dim)
        self.set_setting('use_rfb', use_rfb)
        super().__init__()
        
        self.reduction_conv = nn.Conv2d(patch_emb_dim, hidden_dim // 4, 1)
        self.norm = nn.BatchNorm2d(hidden_dim // 4)
        self.act = nn.ReLU(True)
        self.attention_score_conv = nn.Conv2d(hidden_dim, 4, 1)
        
        self.print_settings()
        
    def forward(self, features_list):
        features = []
        for feat in features_list:
            features.append(self.act(self.norm(self.reduction_conv(feat))))
        x = torch.cat(features, 1)
        weights = self.attention_score_conv(x).softmax(1) 
        
        for i, feat in enumerate(features_list):
            if i == 0:
                out = features_list[0] * weights[:, 0].unsqueeze(1)
            else:
                out = out + feat * weights[:, i].unsqueeze(1)
        
        return out    
    
    
    
class ReweighAccumulator2(BaseModel):
    
    def __init__(self, patch_emb_dim, hidden_dim, use_rfb=True):
        self.set_setting('patch_emb_dim', patch_emb_dim)
        self.set_setting('hidden_dim', hidden_dim)
        self.set_setting('use_rfb', use_rfb)
        super().__init__()
        
        self.shared_in_layers = nn.Sequential(
            nn.Conv2d(patch_emb_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )
        
        self.shared_attn_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(True)
        )
        self.attention_score_conv = nn.Conv2d(hidden_dim, 4, 1)
        self.act = nn.ReLU(True)
        
        self.out_conv = nn.Conv2d(hidden_dim, patch_emb_dim, 1)
        self.print_settings()
        
    def forward(self, features_list):
        features = []
        for feat in features_list:
            features.append(self.shared_in_layers(feat))
        
        attn_features = []
        for feat in features:
            attn_features.append(self.shared_attn_layer(feat))
        attn = self.attention_score_conv(torch.cat(attn_features, 1)).softmax(1)
        
        # Additive combination
        for i, feat in enumerate(features):
            if i == 0:
                out = feat * attn[:, 0].unsqueeze(1)
            else:
                out = out + feat * attn[:, i].unsqueeze(1)
        
        out1 = sum([feat for feat in features])
        out = self.out_conv(self.act(out + out1))
        
        return out    
    

class ReweighAccumulator3(BaseModel):
    
    def __init__(self, patch_emb_dim, hidden_dim, use_rfb=True):
        self.set_setting('patch_emb_dim', patch_emb_dim)
        self.set_setting('hidden_dim', hidden_dim)
        self.set_setting('use_rfb', use_rfb)
        super().__init__()
        
        self.reduction_conv = nn.Conv2d(patch_emb_dim, hidden_dim // 4, 1)
        self.norm = nn.BatchNorm2d(hidden_dim // 4)
        self.act = nn.ReLU(True)
        self.attention_score_conv = nn.Conv2d(hidden_dim, 4, 1)
        
        self.print_settings()
        
    def forward(self, features_list):
        
        out = sum([feat for feat in features_list])
        
        features = []
        for feat in features_list:
            features.append(self.act(self.norm(self.reduction_conv(feat))))
        x = torch.cat(features, 1)
        weights = self.attention_score_conv(x).softmax(1) 
        
        for i, feat in enumerate(features_list):
            if i == 0:
                out1 = features_list[0] * weights[:, 0].unsqueeze(1)
            else:
                out1 = out1 + feat * weights[:, i].unsqueeze(1)
        
        return out + out1 

    
    
class SAAccumulator(BaseModel):
    
    def __init__(self, patch_emb_dim, hidden_dim, use_rfb=True,
                 num_heads=2):
        self.set_setting('patch_emb_dim', patch_emb_dim)
        self.set_setting('hidden_dim', hidden_dim)
        self.set_setting('use_rfb', use_rfb)
        self.set_setting('num_heads', num_heads)
        super().__init__()
        
        self.in_layers = nn.Sequential(
            nn.Conv2d(patch_emb_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # RFB_modified(
            #     hidden_dim, 
            #     hidden_dim) if self.use_rfb else nn.Identity(),
            # nn.Conv2d(hidden_dim, patch_emb_dim, 1)
        )
        
        self.positional_embedding = nn.Parameter(Attention.sinusoidal_embedding(
            4, hidden_dim
        ), requires_grad=False)
        self.patch_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim),
                                            requires_grad=True)
        nn.init.trunc_normal_(self.patch_embedding, std=0.2)
        
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = Attention(dim=hidden_dim, 
                                   num_heads=num_heads,
                                   attention_dropout=0, 
                                   projection_dropout=0)
        
        self.reduction_conv = nn.Conv2d(hidden_dim, hidden_dim // 4, 1)
        self.act = nn.ReLU(True)
        self.attention_score_conv = nn.Conv2d(hidden_dim // 4, hidden_dim, 1)
        
        self.out = nn.Conv2d(hidden_dim, patch_emb_dim, 1)
        
        self.print_settings()
        
    def forward(self, features_list):
        features = []
        for feat in features_list:
            features.append(self.in_layers(feat))
        
        t_x = torch.stack(features, 1)  # B x 4 x C x H/32 x W/32
        H, W = t_x.shape[-2:]
        t_x = rearrange(t_x, 'B scales C H W -> (B H W) scales C')
        
        # Add spatial embeddings
        t_x = t_x + self.positional_embedding
        
        B = t_x.shape[0]
        t_x = torch.cat([repeat(self.patch_embedding, '1 1 d -> b 1 d', b=B), 
                         t_x], 1)
        
        # Attention forward and get patch embeddings
        t_out = self.self_attn(self.pre_norm(t_x))  
        out = rearrange(t_out[:, 0], '(B H W) C -> B C H W', H=H, W=W)
        out = self.out(out)
        return out
    
    
    
# ========================================================================== #
# * ### * ### * ### *        Shape Mapper Modules        * ### * ### * ### * #
# ========================================================================== #


class NaiveShapeMapper2d(BaseModel):
    
    def __init__(
            self, 
            hidden_channels,
            patch_emb_channels,
            downsample_mode='max_pool',
            stage_dims=[64, 128, 256, 512],
            block_expansion=4
            ):
        super().__init__()
        
        self.set_setting('stage_dims', stage_dims)
        self.set_setting('block_expansion', block_expansion)
        self.set_setting('hidden_channels', hidden_channels)
        self.set_setting('patch_emb_channels', patch_emb_channels)
        
        self.stage1_processor = nn.Sequential(
            nn.Conv2d(
                self.stage_dims[0] * self.block_expansion, 
                hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage2_processor = nn.Sequential(
            nn.Conv2d(
                self.stage_dims[1] * self.block_expansion, 
                hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage3_processor = nn.Sequential(
            nn.Conv2d(
                self.stage_dims[2] * self.block_expansion, 
                hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage4_processor = nn.Sequential(
            nn.Conv2d(
                self.stage_dims[3] * self.block_expansion, 
                hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        self.accumulator = nn.Sequential(
            nn.Conv2d(4 * patch_emb_channels, patch_emb_channels, 1)
        )
        self.print_settings()
        
    def forward(self, features_list):
        # Shape mapping
        x4, x8, x16, x32 = features_list
                
        emb_s4 = self.stage1_processor(x4)
        emb_s8 = self.stage2_processor(x8)
        emb_s16 = self.stage3_processor(x16)
        emb_s32 = self.stage4_processor(x32)
        
        feats =  [emb_s4, emb_s8, emb_s16, emb_s32]
        
        # Accumulate
        patch_embeddings = self.accumulator(torch.cat(feats, 1))
        
        return patch_embeddings, \
               rearrange(patch_embeddings, 'B C H W -> B (H W) C')


class StageSeparatedShapeMapper2d(BaseModel):
    
    def __init__(
            self, 
            hidden_channels,
            patch_emb_channels,
            downsample_mode='max_pool',
            stage_dims=[64, 128, 256, 512],
            block_expansion=4
            ):
        super().__init__()
        
        self.set_setting('stage_dims', stage_dims)
        self.set_setting('block_expansion', block_expansion)
        self.set_setting('hidden_channels', hidden_channels)
        self.set_setting('patch_emb_channels', patch_emb_channels)
        
        self.stage1_processor = nn.Sequential(
            RFB(
                self.stage_dims[0] * self.block_expansion, 
                hidden_channels),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage2_processor = nn.Sequential(
            RFB(
                self.stage_dims[1] * self.block_expansion, 
                hidden_channels),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage3_processor = nn.Sequential(
            RFB(
                self.stage_dims[2] * self.block_expansion, 
                hidden_channels),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage4_processor = nn.Sequential(
            RFB(
                self.stage_dims[3] * self.block_expansion, 
                hidden_channels),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        self.accumulator = nn.Sequential(
            RFB(
                4 * patch_emb_channels, 
                patch_emb_channels),
            nn.Conv2d(patch_emb_channels, patch_emb_channels, 1)
        )
        self.print_settings()
        
    def forward(self, features_list):
        # Shape mapping
        x4, x8, x16, x32 = features_list
                
        emb_s4 = self.stage1_processor(x4)
        emb_s8 = self.stage2_processor(x8)
        emb_s16 = self.stage3_processor(x16)
        emb_s32 = self.stage4_processor(x32)
        
        feats =  [emb_s4, emb_s8, emb_s16, emb_s32]
        
        # Accumulate
        patch_embeddings = self.accumulator(torch.cat(feats, 1))
        
        return patch_embeddings, \
               rearrange(patch_embeddings, 'B C H W -> B (H W) C')
               

               
class StageSeparatedShapeMapperLite2d(BaseModel):
    
    def __init__(
            self, 
            hidden_channels,
            patch_emb_channels,
            downsample_mode='max_pool',
            stage_dims=[64, 128, 256, 512],
            block_expansion=4
            ):
        super().__init__()
        
        self.set_setting('stage_dims', stage_dims)
        self.set_setting('block_expansion', block_expansion)
        self.set_setting('hidden_channels', hidden_channels)
        self.set_setting('patch_emb_channels', patch_emb_channels)
        
        self.stage1_processor = nn.Sequential(
            RFB(
                self.stage_dims[0] * self.block_expansion, 
                hidden_channels),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage2_processor = nn.Sequential(
            RFB(
                self.stage_dims[1] * self.block_expansion, 
                hidden_channels),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage3_processor = nn.Sequential(
            RFB(
                self.stage_dims[2] * self.block_expansion, 
                hidden_channels),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        self.stage4_processor = nn.Sequential(
            RFB(
                self.stage_dims[3] * self.block_expansion, 
                hidden_channels),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        self.accumulator = nn.Sequential(
            nn.Conv2d(patch_emb_channels * 4, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            RFB(
                hidden_channels, 
                hidden_channels),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        self.print_settings()
        
    def forward(self, features_list):
        # Shape mapping
        x4, x8, x16, x32 = features_list
                
        emb_s4 = self.stage1_processor(x4)
        emb_s8 = self.stage2_processor(x8)
        emb_s16 = self.stage3_processor(x16)
        emb_s32 = self.stage4_processor(x32)
        
        feats =  [emb_s4, emb_s8, emb_s16, emb_s32]
        
        # Accumulate
        patch_embeddings = self.accumulator(torch.cat(feats, 1))
        
        return patch_embeddings, \
               rearrange(patch_embeddings, 'B C H W -> B (H W) C')
               
               
               
class ShapeMapperV22d(BaseModel):
    
    def __init__(
            self, 
            hidden_channels,
            patch_emb_channels,
            downsample_mode='max_pool',
            stage_dims=[64, 128, 256, 512],
            block_expansion=4,
            use_rfb=True,
            accumulator='conv',
            use_rfb_lite=False
            ):
        super().__init__()
        
        self.set_setting('stage_dims', stage_dims)
        self.set_setting('block_expansion', block_expansion)
        self.set_setting('hidden_channels', hidden_channels)
        self.set_setting('patch_emb_channels', patch_emb_channels)
        self.set_setting('use_rfb', use_rfb)
        self.set_setting('accumulator_name', accumulator)
        self.set_setting('use_rfb_lite', use_rfb_lite)
        
        RFB_modified = RFB
        if self.use_rfb_lite:
            RFB_modified = RFB_lite 
        
        
        s1_layer1 = RFB_modified(self.stage_dims[0] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[0] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage1_processor = nn.Sequential(
            s1_layer1,
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        s2_layer1 = RFB_modified(self.stage_dims[1] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[1] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage2_processor = nn.Sequential(
            s2_layer1,
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        s3_layer1 = RFB_modified(self.stage_dims[2] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[2] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage3_processor = nn.Sequential(
            s3_layer1,
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        s4_layer1 = RFB_modified(self.stage_dims[3] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[3] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage4_processor = nn.Sequential(
            s4_layer1,
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        if 'conv' in self.accumulator_name.lower():
            self.accumulator = ConvAccumulator(patch_emb_channels, 
                                               hidden_channels,
                                               use_rfb=self.use_rfb)
        elif 'weigh' in self.accumulator_name.lower():
            if self.accumulator_name.lower() == 'weigh2':
                self.accumulator = ReweighAccumulator2(patch_emb_channels,
                                                       hidden_channels,
                                                       use_rfb=self.use_rfb)
            elif self.accumulator_name.lower() == 'weigh3':
                self.accumulator = ReweighAccumulator3(patch_emb_channels,
                                                       hidden_channels,
                                                       use_rfb=self.use_rfb)
            else:
                self.accumulator = ReweighAccumulator(patch_emb_channels,
                                                      hidden_channels,
                                                      use_rfb=self.use_rfb)
        elif 'se' in self.accumulator_name.lower():
            self.accumulator = SEAccumulator(patch_emb_channels,
                                             hidden_channels,
                                             use_rfb=self.use_rfb)
        elif 'sa' in self.accumulator_name.lower():
            self.accumulator = SAAccumulator(patch_emb_channels,
                                             hidden_channels,
                                             use_rfb=self.use_rfb)
            
        self.print_settings()
        
    def forward(self, features_list):
        # Shape mapping
        x4, x8, x16, x32 = features_list
                
        emb_s4 = self.stage1_processor(x4)
        emb_s8 = self.stage2_processor(x8)
        emb_s16 = self.stage3_processor(x16)
        emb_s32 = self.stage4_processor(x32)
        
        feats =  [emb_s4, emb_s8, emb_s16, emb_s32]
        
        # Accumulate
        # patch_embeddings = self.accumulator(torch.cat(feats, 1))
        patch_embeddings = self.accumulator(feats)
        
        return patch_embeddings, \
               rearrange(patch_embeddings, 'B C H W -> B (H W) C')
               
               
               
class ShapeMapperV32d(BaseModel):
    
    def __init__(
            self, 
            hidden_channels,
            patch_emb_channels,
            downsample_mode='max_pool',
            stage_dims=[64, 128, 256, 512],
            block_expansion=4,
            use_rfb=True,
            accumulator='conv',
            use_rfb_lite=False
            ):
        super().__init__()
        
        self.set_setting('stage_dims', stage_dims)
        self.set_setting('block_expansion', block_expansion)
        self.set_setting('hidden_channels', hidden_channels)
        self.set_setting('patch_emb_channels', patch_emb_channels)
        self.set_setting('use_rfb', use_rfb)
        self.set_setting('accumulator_name', accumulator)
        self.set_setting('use_rfb_lite', use_rfb_lite)
        
        RFB_modified = RFB
        if self.use_rfb_lite:
            RFB_modified = RFB_lite 
        
        
        s1_layer1 = RFB_modified(self.stage_dims[0] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[0] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage1_processor_a = nn.Sequential(
            s1_layer1,
        )
        self.stage1_processor_b = nn.Sequential(
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode)
        )
        self.stage1_processor_c = nn.Sequential(
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
            )
        
        s2_layer1 = RFB_modified(self.stage_dims[1] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[1] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage2_processor_a = nn.Sequential(
            s2_layer1,
        )
        self.stage2_processor_b = nn.Sequential(
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode)
        )
        self.stage2_processor_c = nn.Sequential(
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode),
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        s3_layer1 = RFB_modified(self.stage_dims[2] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[2] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage3_processor_a = nn.Sequential(
            s3_layer1,
        )
        self.stage3_processor_b = nn.Sequential(
            Downsample2x(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                mode=downsample_mode)
        )
        self.stage3_processor_c = nn.Sequential(
            nn.Conv2d(hidden_channels, patch_emb_channels, 1)
        )
        
        s4_layer1 = RFB_modified(self.stage_dims[3] * self.block_expansion, 
                                 hidden_channels) if self.use_rfb else \
                    nn.Sequential(
                       nn.Conv2d(self.stage_dims[3] * self.block_expansion,
                                 hidden_channels, 1, bias=False),
                       nn.BatchNorm2d(hidden_channels),
                       nn.ReLU(True))
        self.stage4_processor_a = nn.Sequential(
            s4_layer1,
        )
        self.stage4_processor_c = nn.Sequential(
            nn.Conv2d(hidden_channels, patch_emb_channels, 1),
        )
        
        if 'conv' in self.accumulator_name.lower():
            self.accumulator = ConvAccumulator(patch_emb_channels, 
                                               hidden_channels,
                                               use_rfb=self.use_rfb)
        elif 'weigh' in self.accumulator_name.lower():
            if self.accumulator_name.lower() == 'weigh2':
                self.accumulator = ReweighAccumulator2(patch_emb_channels,
                                                       hidden_channels,
                                                       use_rfb=self.use_rfb)
            elif self.accumulator_name.lower() == 'weigh3':
                self.accumulator = ReweighAccumulator3(patch_emb_channels,
                                                       hidden_channels,
                                                       use_rfb=self.use_rfb)
            else:
                self.accumulator = ReweighAccumulator(patch_emb_channels,
                                                      hidden_channels,
                                                      use_rfb=self.use_rfb)
        elif 'se' in self.accumulator_name.lower():
            self.accumulator = SEAccumulator(patch_emb_channels,
                                             hidden_channels,
                                             use_rfb=self.use_rfb)
        elif 'sa' in self.accumulator_name.lower():
            self.accumulator = SAAccumulator(patch_emb_channels,
                                             hidden_channels,
                                             use_rfb=self.use_rfb)
            
        self.print_settings()
        
    def forward(self, features_list):
        # Shape mapping
        x4, x8, x16, x32 = features_list
        
        x4_a = self.stage1_processor_a(x4)
        add_s8 = self.stage1_processor_b(x4_a)
        emb_s4 = self.stage1_processor_c(add_s8)
        
        x8_a = self.stage2_processor_a(x8)
        add_s16 = self.stage2_processor_b(x8_a + add_s8)
        emb_s8 = self.stage2_processor_c(add_s16)
        
        x16_a = self.stage3_processor_a(x16)
        add_s32 = self.stage3_processor_b(x16_a + add_s16)
        emb_s16 = self.stage3_processor_c(add_s32)
        
        x32_a = self.stage4_processor_a(x32)
        emb_s32 = self.stage4_processor_c(x32_a + add_s32)
        
        feats =  [emb_s4, emb_s8, emb_s16, emb_s32]
        
        # Accumulate
        # patch_embeddings = self.accumulator(torch.cat(feats, 1))
        patch_embeddings = self.accumulator(feats)
        
        return patch_embeddings, \
               rearrange(patch_embeddings, 'B C H W -> B (H W) C')


    