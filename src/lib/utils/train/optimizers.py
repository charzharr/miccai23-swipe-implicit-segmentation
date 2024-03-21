""" utils/train/optimizers.py (By: Charley Zhang)

Boilerplate for setting up a common experiment component: optimizers.
"""

import torch
import torch.nn as nn

EPS = 1e-08


# ------------------ ##  Optimizers  ## ------------------ #

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
            betas=opt_config.betas,
            eps=EPS
        )
        print(f'💠 Adam optimizer initiated with lr={lr}, wd={wdecay}, \n'
              f'   betas={opt_config.betas}.')
    elif opt == 'adamw':
        opt_config = config.train.optimizer.adamw
        optimizer = torch.optim.AdamW(
            params, 
            lr=lr, 
            weight_decay=wdecay,
            betas=opt_config.betas,
            eps=EPS
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


