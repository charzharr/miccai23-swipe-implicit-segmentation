""" src/experiments/setup.py (By: Charley Zhang)
Boilerplate for setting up common experiment components:
  - Gathers the necessary resources, modules, and utilities.
  - Configures all essential training components 
    (model architecture + params, criterion, optimizer, schedulers)
  - Initializes stat trackers and experiment trackers
"""

import torch
import torch.nn as nn

from lib.utils.train import schedulers



# ------------------ ##  Training Components  ## ------------------ #


def get_criterion(config):
    crit_name = config.train.criterion.name
    if crit_name in config.train.criterion:
        crit_config = config.train.criterion[crit_name]

    # ------ #  multi-D Losses  # ------ #
    if crit_name == 'mse':
        criterion = nn.MSELoss()
    elif crit_name == 'l1':
        criterion = nn.L1Loss()
    elif crit_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    # ------ #  2D Losses  # ------ #
    
    # ------ #  3D Losses  # ------ #
    elif crit_name == 'byol':
        from lib.assess.losses3d import BYOL3d
        criterion = BYOL3d()
    elif crit_name == 'dice-ce_3d':
        from lib.assess.nnunet_loss import (DC_and_CE_loss, SoftDiceLoss,
                                            softmax_helper)
        # criterion = DC_and_CE_loss({'do_bg': False, 'batch_dice': False}, {},
        #                             ignore_label=None)
        criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=False, 
                                 do_bg=False)
        # from lib.assess.losses3d import DiceCrossEntropyLoss3d
        # criterion = DiceCrossEntropyLoss3d(
        #     alpha=crit_config.alpha
        # )
    
    
    else:
        raise ValueError(f"Criterion {crit_name} is not supported.")

    return criterion


def get_scheduler(config, optimizer):
    sched = config.train.scheduler.name
    t = config.train.start_epoch
    T = config.train.epochs
    rampup_rates = config.train.scheduler.rampup_rates
    min_lr = config.train.scheduler.min_lr
    
    if sched == 'uniform':
        scheduler = schedulers.Uniform(
            optimizer,
            rampup_rates=rampup_rates
        )
    elif 'poly' in sched:
        sched_config = config.train.scheduler.poly
        scheduler = schedulers.PolynomialDecay(
            optimizer,
            T,
            t=t,
            power=sched_config.power,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'exponential' in sched:
        sched_config = config.train.scheduler.exponential
        scheduler = schedulers.ExponentialDecay(
            optimizer,
            t=t,
            exp_factor=sched_config.exp_factor,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'linear' in sched:
        sched_config = config.train.scheduler.linear
        scheduler = schedulers.LinearDecay(
            optimizer,
            T=T,
            end_factor=sched_config.end_factor,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'consistencycosine' in sched:  # orig_lr * cos(7*pi*t/(16*T)) 
        scheduler = schedulers.ConsistencyCosineDecay( 
            optimizer,
            T, 
            t=t,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'cosine' in sched:   # 0.5(1 + cos(pi*t/T)) * orig_lr
        scheduler = schedulers.CosineDecay(  
            optimizer,
            T, 
            t=t,
            minlr=min_lr,
            rampup_rates=rampup_rates
        )
    elif 'plateau' in sched:
        sched_config = config.train.scheduler.plateau
        scheduler = schedulers.ReduceOnPlateau(
            optimizer,
            factor=sched_config.factor,
            patience=sched_config.patience,
            lowerbetter=True,
            rampup_rates=rampup_rates
        )
    elif 'step' in sched:
        sched_config = config.train.scheduler.step
        scheduler = schedulers.StepDecay(
            optimizer,
            factor=sched_config.factor,
            T=T,
            steps=sched_config.steps,
            rampup_rates=rampup_rates
        )
    
    return scheduler


def get_optimizer(config, params):
    opt = config.train.optimizer.name
    lr = config.train.optimizer.lr
    wdecay = config.train.optimizer.wt_decay

    if opt == 'adam':
        opt_config = config.train.optimizer.adam
        optimizer = torch.optim.Adam(
            params, 
            lr=lr, 
            weight_decay=wdecay,
            betas=opt_config.betas
        )
        print(f'💠 Adam optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   betas={opt_config.betas}.')
    elif opt == 'adamw':
        opt_config = config.train.optimizer.adamw
        optimizer = torch.optim.AdamW(
            params, 
            lr=lr, 
            weight_decay=wdecay,
            betas=opt_config.betas
        )
        print(f'💠 AdamW optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   betas={opt_config.betas}.')
    elif 'nesterov' in opt:
        opt_config = config.train.optimizer.nesterov
        optimizer = torch.optim.SGD(
            params, 
            lr=lr, 
            momentum=opt_config.momentum, 
            weight_decay=wdecay,
            nesterov=True
        )
        print(f'💠 Nesterov optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   momentum={opt_config.momentum}.')
    elif 'sgd' in opt:  # sgd
        opt_config = config.train.optimizer.sgd
        optimizer = torch.optim.SGD(
            params, 
            lr=lr, 
            momentum=opt_config.momentum, 
            weight_decay=wdecay
        )
        print(f'💠 SGD optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   momentum={opt_config.momentum}.')
    else:
        raise ValueError(f'Optimizer "{opt}" is not supported')

    return optimizer


