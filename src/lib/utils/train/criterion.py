""" utils/train/criterion.py (By: Charley Zhang)

Boilerplate for setting up a common experiment component: loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ ##  Optimizers  ## ------------------ #


def get_criterion(config):
    crit_name = config.train.criterion.name
    if crit_name in config.train.criterion:
        crit_config = config.train.criterion[crit_name]

    # ------ #  multi-D Losses  # ------ #
    if crit_name == 'l1':
        criterion = nn.L1Loss()
    elif crit_name == 'mse' or crit_name == 'l2':
        criterion = nn.MSELoss()
    elif crit_name == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif crit_name == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif crit_name == 'dice':
        from lib.losses.nnunet_utils import softmax_helper, sigmoid_helper
        from lib.losses.nnunet_losses import SoftDiceLoss
        nonlin = softmax_helper if crit_config.nonlin == 'softmax' else \
                 sigmoid_helper
        criterion = SoftDiceLoss(apply_nonlin=nonlin, 
                                 batch_dice=crit_config.batch_dice,
                                 do_bg=crit_config.do_bg,
                                 smooth=crit_config.smooth)
    elif 'dice-bce' in crit_name:
        dc_kw = {} 
        if 'dc_kw' in crit_config:
            dc_kw = crit_config.dc_kw
        ce_kw = {}
        if 'ce_kw' in crit_config:
            ce_kw = crit_config.ce_kw

        from lib.losses.nnunet_losses import DC_and_BCE_loss         
        criterion = DC_and_BCE_loss(
            dc_kw, ce_kw,
            weight_dice=crit_config.dc_weight,
            weight_ce=crit_config.ce_weight
        )
    elif 'dice-ce' in crit_name:
        ce_kw = {}
        if 'ce_kw' in crit_config:
            ce_kw = crit_config.ce_kw
            if 'weight' in ce_kw and ce_kw['weight']:
                ce_kw['weight'] = torch.tensor(ce_kw['weight'])
        dc_kw = {} 
        if 'dc_kw' in crit_config:
            dc_kw = crit_config.dc_kw

        from lib.losses.nnunet_losses import DC_and_CE_loss
        criterion = DC_and_CE_loss(dc_kw, ce_kw, ignore_label=None, 
                                   weight_dice=crit_config.dc_weight,
                                   weight_ce=crit_config.ce_weight)
    
    # ------ #  2D Losses  # ------ #
    
    # ------ #  3D Losses  # ------ #

    else:
        raise ValueError(f"Criterion {crit_name} is not supported.")

    return criterion


