
from ast import Not
import os, pathlib, glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nets.init import init_weights
from collections import OrderedDict


EXP_NAMES = ('miccai23shape',)


def _load_opt_value(cfg, val):
    if val in cfg:
        return cfg[val]
    return None


# ========================================================================== #
# * ### * ### * ### *              Main API              * ### * ### * ### * #
# ========================================================================== #


def get_model_components(config):
    """ Main API: Called from run_experiment to retrieve experiment model. """
    init = None
    
    cp_name = None
    if 'run_name' in config.experiment.checkpoint:
        cp_name = config.experiment.checkpoint.run_name
    
    if cp_name:
        cp_d = get_checkpoint_components(config, cp_name)
        cp_config = cp_d['config']
        model = modify_pretrain_model_for_finetune(cp_d['model'], 
                                                   cp_d['config'], 
                                                   config)
    else:
        # from experiments.model3d_backbones import get_model
        model = get_model(config, init=init)
        cp_config = None
    
    return {
        'model': model,
        'cp_config': cp_config
    }
    

def get_model(config, init=None):
    """
    Creates model instances and optionally initializes paramter weights, and
    adds last layer predictive class-balancing.
    Args:
        config: experiment-specific config with model instantiation information
        init: 'norma', 'xavier', 'kaiming', 'orthogonal'
    """
    
    model_config = _load_opt_value(config.model, config.model.name)
    
    dataset_config = config.data[config.data.name]
    in_channels = _load_opt_value(dataset_config, 'net_in_channels')
    in_size = _load_opt_value(dataset_config, 'net_in_size')
    out_channels = _load_opt_value(dataset_config, 'net_out_channels')

    # Get network normalize and activation layer factories
    from lib.nets.component_factories import NormFactory3d, NormFactory2d
    norm_cfg = config.model.norm
    if '2d' in config.model.name:
        norm_factory = NormFactory2d(norm_cfg.name, groups=norm_cfg.groups)
    else:
        assert '3d' in config.model.name
        norm_factory = NormFactory3d(norm_cfg.name, groups=norm_cfg.groups)
    print(f'* Norm:', norm_factory)
    
    from lib.nets.component_factories import ActFactory
    act_cfg = config.model.act
    act_factory = ActFactory(act_cfg.name)
    print(f'* Act:', act_factory)
    
        
    # --- Experiment Adjustments --- #
    if _load_opt_value(config.model, 'input_location_channels'):
        in_channels += 3
    
    # --- Get Model(s) --- #
    if config.model.name == 'unet3d':
        from lib.nets.volumetric.unet3d import UNet3d
        model = UNet3d(in_channels, 
                       out_channels,
                       base_channels=model_config.base_dim,
                       bilinear=model_config.bilinear)
    elif config.model.name == 'unet2d':
        from lib.nets.planar.unet2d import UNet2d
        model = UNet2d(in_channels, 
                       out_channels,
                       norm_factory, 
                       act_factory,
                       base=32,
                       bilinear=model_config.bilinear) 
    elif 'resnet' in config.model.name or 'resunet' in config.model.name:
        model = get_resnets(config.model.name,
                            in_channels, 
                            out_channels, 
                            norm_factory, 
                            act_factory,
                            config)
    elif 'pranet' in config.model.name:
        from .models.pranet.pranet import PraNet2d
        model = PraNet2d(config, act_factory, norm_factory,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         channel=model_config.feature_size,
                         pretrained=model_config.pretrained)
    elif 'unetrpp' in config.model.name:
        if '3d' in config.model.name:
            from .models.unetr.unetr3d_pp import UNETR_PP
        else:
            from .models.unetr.unetr2d_pp import UNETR_PP
        model = UNETR_PP(in_channels=in_channels,
                         out_channels=out_channels,
                         img_size=in_size,
                         feature_size=model_config.feature_size,
                         num_heads=model_config.num_heads,
                         depths=model_config.depths,
                         dims=model_config.dims,
                         do_ds=config.train.deep_sup)
    elif 'iosnet' in config.model.name:
        model = get_iosnet(config,
                           in_channels, 
                           out_channels,
                           norm_factory, 
                           act_factory)
    elif 'ours' in config.model.name:
        model = get_ours(config, 
                         in_size,
                         in_channels,
                         out_channels,
                         norm_factory, 
                         act_factory)
    else:
        raise NotImplementedError(config.model.name)
    
    # --- Initialize Weights --- #
    if init:
        print('\t🧠  Model: Initializing model weights {init}')
        model = init_weights(model, init_type=init)
        
    # --- Print Model Info --- #
    print('\t🧠  Model Architecture:\n', model)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'\t🧠  Model: Model retrieved with {num_params:,} parameters.')
    
    ## Get Error: Floating point exception (core dumped)
    # from lib.nets.utils import get_model_flops
    # input_shape = [1, in_channels] + list(in_size)
    # input_shape = [1, 1, 96, 96, 96]
    # print(f'\t🧠  Model: Analyzing FLOPS + MAdds..')
    # get_model_flops(model, input_shape, disp=True)
    
    return model


# ========================================================================== #
# * ### * ### * ### *            Get Backbones           * ### * ### * ### * #
# ========================================================================== #



def get_ours(config, in_size, in_channels, out_channels, 
             norm_factory, act_factory):
    data_config = config.data[config.data.name]
    model_config = config.model[config.model.name]
    
    out_channels = config.data[config.data.name].num_classes
    out_channels = 1 if out_channels == 2 else out_channels
    ndim = int(config.model.name[-2])
    
    ## Get Encoder
    if 'cct' in model_config.enc_name:  # replace True when ready
        if ndim == 2:
            from .models.ours.vit.cct import CCT 
        elif ndim == 3:
            from .models.ours.vit.cct3d import CCT
        emb_dim = model_config.embedding_dim
        encoder = CCT(
            img_size=in_size,
            embedding_dim=emb_dim,
            n_conv_layers=model_config.n_conv_layers,
            n_input_channels=in_channels, 
            n_output_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            mlp_ratio=model_config.mlp_ratio,
            positional_embedding=model_config.pos_embedding,
            seq_pool=model_config.seq_pool,
            attention_dropout=model_config.dropout_attn,
            stochastic_depth=model_config.dropout_depth,
        )
        
        emb_dim = model_config.embedding_dim
        dec_feats = ndim * sum(model_config.feature_flags[0:3:2])   # 1st and 3rd flag
        dec_feats += emb_dim * sum(model_config.feature_flags[1:4:2]) # 2nd and 4th
        dec_feats += emb_dim * sum(model_config.feature_flags[4:])
        
        glob_dec_feats = emb_dim + ndim
    else:
        if 'dec_separate_classes' in model_config and model_config.dec_separate_classes:
            assert '3d' in model_config.enc_name, f'{model_config.enc_name}'
            from .models.ours.v3 import backbone3d
            backbone3d.BLOCK_EXPANSION = model_config.block_expansion
            backbone3d.STAGE_DIMS = model_config.stage_dims
            backbone_init = backbone3d.ResNet3d
            
            from .models.ours.v3.backbone3d import Bottle2neck, Conv3d
            block = Bottle2neck
            conv_layer = Conv3d
        else:
            if '2d' in model_config.enc_name:
                from .models.ours.v2 import backbone2d
                backbone2d.BLOCK_EXPANSION = model_config.block_expansion
                backbone2d.STAGE_DIMS = model_config.stage_dims
                backbone_init = backbone2d.ResNet2d
                
                from .models.ours.v2.backbone2d import Bottle2neck, Conv2d
                block = Bottle2neck
                conv_layer = Conv2d
            else:
                assert '3d' in model_config.enc_name, f'{model_config.enc_name}'
                from .models.ours.v2 import backbone3d
                backbone3d.BLOCK_EXPANSION = model_config.block_expansion
                backbone3d.STAGE_DIMS = model_config.stage_dims
                backbone_init = backbone3d.ResNet3d
                
                from .models.ours.v2.backbone3d import Bottle2neck, Conv3d
                block = Bottle2neck
                conv_layer = Conv3d
        
        print(f'\t🧠  Block: {block}')
    
        # Get Layers Configuration
        if model_config.layers == 28:
            layers = [2, 2, 2, 2]
        elif model_config.layers == 50:  # 28w_4s
            layers = [3, 4, 6, 3]
        elif model_config.layers == 14:
            layers = [1, 1, 1, 1]
    
        # Get Backbone
        encoder = backbone_init(
            config,
            block, 
            layers, 
            norm_factory, 
            act_factory, 
            conv_layer=conv_layer,                  
            baseWidth=model_config.base_width,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=model_config.dropout,
            stochastic_depth=model_config.stochastic_depth,
            deep_sup=False
        )
        
        feat_flags = model_config.feature_flags
        emb_dim = model_config.embedding_dim
        glob_emb_dim = emb_dim
        if 'glob_embedding_dim' in model_config:
            glob_emb_dim = model_config.glob_embedding_dim
        
        dec_feats = ndim * sum(feat_flags[0:3:2])     # pos, 1st and 3rd flag
        dec_feats += glob_emb_dim if feat_flags[1] else 0
        dec_feats += emb_dim if feat_flags[3] else 0  # local embedding
        loc_fourier = _load_opt_value(model_config, 'local_fourier_coords_dim')
        glo_fourier = _load_opt_value(model_config, 'global_fourier_coords_dim')
        dec_feats += loc_fourier if loc_fourier else 0 
        dec_feats += glo_fourier if glo_fourier else 0 
        glob_coords = _load_opt_value(model_config, 'dec_global_coords')
        dec_feats += ndim if glob_coords else 0
        
        glob_dec_feats = glob_emb_dim + ndim
       
    ## Get Decoder
    if 'mlp' in model_config.dec_name:
        from .models.ours.decoders import MLPDecoder
        decoder = MLPDecoder(
            in_channels=dec_feats, 
            out_channels=out_channels,
            hidden_dims=model_config.dec_dims
        )
    elif 'deepsdf' in model_config.dec_name:
        if 'dec_separate_classes' in model_config and model_config.dec_separate_classes:
            from .models.ours.decoders import MCDeepSDFDecoder
            decoder = MCDeepSDFDecoder(
                in_channels=dec_feats, 
                out_channels=out_channels,
                hidden_dims=model_config.dec_dims,
                residual_layers=model_config.dec_residual_layers,
                dropout=model_config.dec_dropout,
                norm='bn',   # 'bn' 'ln' 'none' 'wn'
                d=emb_dim,
            )
        else:
            from .models.ours.decoders import DeepSDFDecoder
            decoder = DeepSDFDecoder(
                in_channels=dec_feats, 
                out_channels=out_channels,
                hidden_dims=model_config.dec_dims,
                residual_layers=model_config.dec_residual_layers,
                dropout=model_config.dec_dropout,
                norm='bn'   # 'bn' 'ln' 'none' 'wn'
            )
        
    global_decoder = None
    if model_config.glob_dec_name and model_config.glob_dec_name != 'none':
        if 'deepsdf' in model_config.dec_name:
            if 'dec_separate_classes' in model_config and model_config.dec_separate_classes:
                from .models.ours.decoders import MCDeepSDFDecoder
                global_decoder = MCDeepSDFDecoder(
                    in_channels=glob_dec_feats, 
                    out_channels=out_channels,
                    hidden_dims=model_config.glob_dec_dims,
                    residual_layers=model_config.glob_dec_residual_layers,
                    dropout=model_config.glob_dec_dropout,
                    norm='bn',   # 'bn' 'ln' 'none' 'wn'
                    d=glob_emb_dim,
                )
            else:
                from .models.ours.decoders import DeepSDFDecoder
                global_decoder = DeepSDFDecoder(
                    in_channels=glob_dec_feats, 
                    out_channels=out_channels,
                    hidden_dims=model_config.glob_dec_dims,
                    residual_layers=model_config.glob_dec_residual_layers,
                    dropout=model_config.glob_dec_dropout,
                    norm='bn'   # 'bn' 'ln' 'none' 'wn'
                )
        else:
            from .models.ours.decoders import MLPDecoder
            global_decoder = MLPDecoder(
                in_channels=glob_dec_feats, 
                out_channels=out_channels,
                hidden_dims=model_config.glob_dec_dims
            )
    
    ## Wrapper
    if ndim == 2:
        from .models.ours.ours_patch import PatchModel2d
        model = PatchModel2d(config, encoder, decoder, 
                             global_decoder=global_decoder)
    else:
        assert ndim == 3
        from .models.ours.ours_patch3d import PatchModel3d
        model = PatchModel3d(config, encoder, decoder, 
                             global_decoder=global_decoder)
    
    return model


def get_iosnet(config, in_channels, out_channels, norm_factory, act_factory):
    assert 'iosnet' in config.model.name
    model_config = config.model[config.model.name]
    ndim = int(config.model.name[-2])
    
    if '2d' in model_config.enc_name:
        from .models.resnets import resnet2d
        resnet2d.BLOCK_EXPANSION = model_config.block_expansion
        resnet2d.STAGE_DIMS = model_config.stage_dims
        num_classes = 1
    else:
        from .models.resnets import resnet3d
        resnet3d.BLOCK_EXPANSION = model_config.block_expansion
        resnet3d.STAGE_DIMS = model_config.stage_dims
        num_classes = out_channels
    
    encoder = get_resnets(model_config.enc_name,
                          in_channels, 
                          out_channels, 
                          norm_factory, 
                          act_factory,
                          config)
    
    # Get Decoder and compute input channels
    from .models.iosnet.decoders import IOSDecoder, MLPDecoder
    num_displacements = 1 if model_config.point_displacement == 0 else \
                        1 + len(model_config.displacements)
    
    feat_flags = model_config.feature_flags
    idim = in_channels if feat_flags[0] else 0
    pdim = ndim if model_config.dec_input_coords else 0  # num coord dims
    feats = 64 + pdim + idim if feat_flags[1] else pdim + idim
    feats += sum([encoder.stage_dims[min(i, 3)] for i in range(0, 5) 
                  if feat_flags[i+2]]) * encoder.block.expansion
    if 'dec_global_coords' in model_config:
        feats += ndim if model_config.dec_global_coords else 0
    decoder = IOSDecoder(
        in_channels=feats, 
        out_channels=num_classes,
        hidden_dims=model_config.dec_dims
    )
    
    if ndim == 2:
        from .models.iosnet.iosnet import IOSNet2d
        model = IOSNet2d(config, encoder, decoder, 
                         displacement=model_config.point_displacement)
    else:
        assert ndim == 3, f'{ndim}'
        from .models.iosnet.iosnet3d import IOSNet3d
        model = IOSNet3d(config, encoder, decoder, 
                         displacement=model_config.point_displacement)
        
    return model


def get_resnets(
        model_name,
        in_channels, 
        out_channels, 
        norm_factory, 
        act_factory,
        config
        ):
    """ Get ResNet variants including:
         - ResNet3d
         - ResUNet3d
         - ResNet2d
         - ResUNet2d
          etc..
    """
    
    model_config = config.model[model_name]
    if model_name == 'resunet3d':
        from .models.resnets import resunet3d
        resunet3d.BLOCK_EXPANSION = model_config.block_expansion
        resunet3d.STAGE_DIMS = model_config.stage_dims
        
        from .models.resnets.resunet3d import (
            ResUNet3d, Bottleneck, Bottle2neck, BottleneckSE, Conv3d
        )
        model_init = ResUNet3d
        conv_layer = Conv3d
    elif model_name == 'resunet2d':
        from .models.resnets import resunet2d
        resunet2d.BLOCK_EXPANSION = model_config.block_expansion
        resunet2d.STAGE_DIMS = model_config.stage_dims
                
        from .models.resnets.resunet2d import (
            ResUNet2d, Bottleneck, Bottle2neck, BottleneckSE, Conv2d
        )
        model_init = ResUNet2d
        conv_layer = Conv2d
    elif model_name == 'resunet_attn2d':
        from .models.resnets import resunet_attn2d
        resunet_attn2d.BLOCK_EXPANSION = model_config.block_expansion
        resunet_attn2d.STAGE_DIMS = model_config.stage_dims
                
        from .models.resnets.resunet_attn2d import (
            ResUNetAttn2d, Bottleneck, Bottle2neck, BottleneckSE, Conv2d
        )
        model_init = ResUNetAttn2d
        conv_layer = Conv2d
    elif model_name == 'resnet2d':
        from .models.resnets.resnet2d import (
            ResNet2d, Bottleneck, Bottle2neck, BottleneckSE, Conv2d
        )
        model_init = ResNet2d
        conv_layer = Conv2d
    elif model_name == 'resnet3d':
        from .models.resnets.resnet3d import (
            ResNet3d, Bottleneck, Bottle2neck, BottleneckSE, Conv3d
        )
        model_init = ResNet3d
        conv_layer = Conv3d
    else:
        raise NotImplementedError(model_name)
    
    # Get Stage Blocks
    block_name = model_config.block 
    if block_name == 'bottleneckSE':
        block = BottleneckSE
    elif block_name == 'bottleneck2':
        block = Bottle2neck
    else:
        block = Bottleneck 
    print(f'\t🧠  Block: {block}, Config: {block_name}')
    
    # Get Layers Configuration
    if model_config.layers == 14:
        layers = [1, 1, 1, 1]
    elif model_config.layers == 28:
        layers = [2, 2, 2, 2]
    elif model_config.layers == 50:  # 28w_4s
        layers = [3, 4, 6, 3]
    elif model_config.layers == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError(f'Resnet with {model_config.layers} layers invalid!')
    
    # Get Backbone
    backbone = model_init(
        block, 
        layers, 
        norm_factory, 
        act_factory, 
        conv_layer=conv_layer,                  
        baseWidth=model_config.base_width,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=model_config.dropout,
        stochastic_depth=model_config.stochastic_depth,
        reduce_conv1_dstride=model_config.reduce_conv1_dstride,
        deep_sup=False
    )
    
    return backbone
    


# ========================================================================== #
# * ### * ### * ### *               Helpers              * ### * ### * ### * #
# ========================================================================== #


def get_checkpoint_components(config, run_name):     
    curr_path = pathlib.Path(__file__).parent.absolute()
    
    exp_path, experiment_name = None, None
    for exp_name in EXP_NAMES:
        tentative_exp_path = curr_path.parent/exp_name/'artifacts'/run_name
        if tentative_exp_path.exists():
            exp_path = tentative_exp_path
            experiment_name = exp_name
    assert exp_path.exists(), f'"run_name" invalid: {run_name}'
    
    print(f"🪧  Loading model checkpoint from experiment {experiment_name} \n"
          f"     Run: {run_name}")
    
    checkpt_dir = exp_path
    assert checkpt_dir.exists(), f'Checkpoint run invalid: {checkpt_dir}'
    all_pth_files = glob.glob(str(checkpt_dir / '*.pth'))

    pth_keyword = str(config.experiment.checkpoint.pth_keyword)
    pth_files = []
    for pth_file in all_pth_files:
        if pth_keyword in pth_file:
            pth_files.insert(0, pth_file)
            break
        pth_files.append(pth_file)
    checkpt_path = pth_files[0]
    
    # Load Checkpoint & Get Original Model
    print(f'\t⭐ Reading from Checkpoint {str(checkpt_path)}')
    checkpoint = torch.load(str(checkpt_path), map_location='cpu')
    cp_config = checkpoint['config']
    cp_state_dict = checkpoint['state_dict']
    if 'ema_state_dict' in checkpoint and \
            'use_ema' in config.experiment.checkpoint and \
            config.experiment.checkpoint.use_ema:
        print(f' * Loading in ema_state_dict')
        cp_state_dict = checkpoint['ema_state_dict']
    else:
        print(f' * Using non-ema model.')
    
    cp_model_module = __import__(f'experiments.{experiment_name}.model_setup')
    cp_get_model = getattr(cp_model_module, experiment_name).model_setup
    cp_model_d = cp_get_model.get_model_components(cp_config)
    
    cp_model = cp_model_d['model']
    
    print(cp_model.load_state_dict(cp_state_dict, strict=False))
    
    return {
        'model': cp_model,
        'config': cp_config
    }
    

def modify_pretrain_model_for_finetune(cp_model, cp_config, ft_config):
    
    return cp_model
    
    # Modify Model for Fine-tuning
    cp_task_name = cp_config.tasks.name
    cp_task_config = cp_config.tasks[cp_task_name]
    out_dim = ft_config.data[ft_config.data.name].num_classes
    if cp_task_name in ('mg', 'rubik'):
        m = nn.Conv3d(32, out_dim, 1, bias=True)
        nn.init.constant_(m.bias, 0)
        cp_model.final_conv = m
        final_model = cp_model
        print(f"🪧  (FT Model) Replaced 'final_conv' w {out_dim} out dims.")
    elif cp_task_name == 'sar': 
        final_model = cp_model.backbone
        m = nn.Conv3d(32, out_dim, 1, bias=True)
        nn.init.constant_(m.bias, 0)
        final_model.final_conv = m
        print(f"🪧  (SAR -> FT) Replaced 'final_conv' w {out_dim} out dims.")

    elif cp_task_name in ('pgl', 'moco', 'ours'):
        final_model = cp_model
        m = nn.Conv3d(32, out_dim, 1, bias=True)
        nn.init.constant_(m.bias, 0)
        final_model.final_conv = m
        print(f"🪧  (MoCo/PGL -> FT) Replaced 'final_conv' w {out_dim} out dims.")
    else:
        raise NotImplementedError(f'Task {cp_task_name} not implemented.')

    return final_model




