""" 
Is called by shell scripts to run experiments.
Main job: call appropriate experiment main.py file and pass on env args.
    (1) Get config file & validate settings
    (2) Parse GPU device info
    (3) Set experiment seed
    (4) Run experiment via the corresponding emain.py

Updates
-------
(2020.11)
  - Added negative seeds for random seed sampling.
  - Added sun grid array job submissions. 
  - Improved exception catching in main experiment to encompass all exceptions.
(2022.01)
  - Added automatic run_experiment module loading. 
  - Moved cfg parsing to run_experiment (run-specific granularity).
  - Added separate get_config file for easy notebook access.
"""

import sys, os
import pathlib
import signal
import re
import math, random
import importlib.util
import click
import copy

import numpy as np
import torch

import lib
from lib.utils.train.configs import get_config

# Some tweaks
torch.set_printoptions(sci_mode=False)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)},
                    suppress=True,
                    precision=3,
                    linewidth=150)

# Automated loading of experiment modules 'run_experiment.py'
src_path = pathlib.Path(__file__).parent.absolute()
exps_path = src_path / 'experiments'
exp_dirs = [d for d in os.listdir(exps_path) if
                (exps_path / d).is_dir() and
                'run_experiment.py' in os.listdir(exps_path / d)]

EXPERIMENTS = {}  # maps experiment names to its run module
for exp_dir in exp_dirs:
    full_dir_path = exps_path / exp_dir
    mod_name = f'experiments.{exp_dir}.run_experiment'
    exps_mod = __import__(mod_name)
    EXPERIMENTS[exp_dir] = getattr(exps_mod, exp_dir).run_experiment


# ========================================================================== #
# * ### * ### * ### *           CLI Start Point          * ### * ### * ### * #
# ========================================================================== #

@click.command()
@click.option('--config', required=True, type=click.Path(exists=False))
@click.option('-ddp', '--distributed', is_flag=True, 
              help='Flag to indicate whether to use distributed training.')
def run_cli(config, distributed): 
    main(config, distributed)


def main(config_name, distributed):
    orig_config = get_experiment_config(config_name, verbose=True)
    
    num_runs = 1 if orig_config.experiment.debug.mode else orig_config.experiment.num_runs
    for run_number in range(1, num_runs + 1):
        config = copy.deepcopy(orig_config)
        exp_id = config.experiment.id 
        matched = re.findall(r'r\d+', exp_id)
        if matched:
            config.experiment.id = exp_id.replace(matched[-1], f'r{run_number}')
        else:
            config.experiment.id = exp_id + f'_r{run_number}'
            
        run(config, distributed)
        

def run(config, distributed):
    cfg = config
    
    # --- ##  GPU device parsing and distributed training init  ## --- #    
    gpu_indices = []
    if torch.cuda.device_count():
        cuda_dev_properties = torch.cuda.get_device_properties(0)
        dev_name = cuda_dev_properties.name
        if 'v100' in dev_name.lower() or 'a10' in dev_name.lower():
            cfg.experiment.amp = True
            print(f'[GPUs] V100 detected! Turning on torch.amp.')
        else:
            cfg.experiment.amp = False
            print(f'[GPUs] No Volta or Ampere detected! Not using torch.amp.')
        print(f'       Current main device: {cuda_dev_properties}')
        gpu_indices = tuple(range(torch.cuda.device_count()))
    cfg.experiment.gpu_idxs = gpu_indices

    if distributed and len(gpu_indices) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'              
        os.environ['MASTER_PORT'] = '8888'
        cfg.experiment.distributed = True
    else:
        cfg.experiment.distributed = False
        cfg.experiment.rank = 0
        device = f'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.experiment.device = device
        print(f'[GPUs] Using device(s) with id(s): {gpu_indices}.')
    
    # --- ##  Final setup  ## --- #
    
    experiment_main = EXPERIMENTS[cfg['experiment']['name']]
    exp_run_args = []
    
    # Run within exception wrapper so processes can end gracefully
    try:
        if cfg.experiment.distributed:
            print(f'[RUN] Using distributed. Spawning {len(gpu_indices)} '
                  'processes.')
            spawn_args = tuple([cfg] + exp_run_args)
            torch.multiprocessing.spawn(
                experiment_main.run, 
                args=spawn_args,
                nprocs=len(gpu_indices))
        else:
            experiment_main.run(0, cfg, *exp_run_args)  # rank 0
    except BaseException as err:
        print(f'\nException thrown:\n', '-' * 30, f'\n{err}', sep='')
        print(f'\nTraceback:\n', '-' * 30, sep='')
        import traceback
        traceback.print_exc()

        print('\n\n' + '*' * 80 + '\n[END] Program Exit Cleanup Initiated!\n')
        kill_children()
        if cfg.experiment.distributed:
            torch.distributed.destroy_process_group()
    finally:
        print('🛑 Ended 🛑\n')


def get_experiment_config(config, merge_default=False, verbose=False):
    """
    Args:
        config (str): config file name
    """
    from lib.utils.train.configs import get_config
    
    # --- ##  Get experiment configuration  ## --- #
    given_cfg_path = pathlib.Path(config)
    if given_cfg_path.exists():
        print(f'[CFG] Given cfg file "{str(given_cfg_path)}" exists! Loading..')
        cfg = get_config(config, merge_default=False, search_dir='')
    else:
        print(f'[CFG] Given cfg file "{str(given_cfg_path)}" does not exist! '
              'Searching for matching name in experiment\'s config folder..')
        curr_path = pathlib.Path(__file__).parent.absolute()
        exp_cfg_path = None
        for exp in EXPERIMENTS:
            if exp in given_cfg_path.name:
                exp_cfg_path = str(curr_path / 'experiments' / exp / 'configs')
                print(f' ✔ Cfg experiment matched at {exp_cfg_path}.')
                break

        if not exp_cfg_path:
            msg = (f'Given config file "{config}" does not contain any of the '
                   f'experiment names in them: {list(EXPERIMENTS.keys())}')
            raise ValueError(msg)
        cfg = get_config(config, merge_default=merge_default, 
                         search_dir=exp_cfg_path)
        
    if verbose:
        from pprint import pprint
        print('* Successfully loaded config:')
        pprint(cfg)
        
    return cfg
        

# ========================================================================== #
# * ### * ### * ### *        Run Setup & Handling        * ### * ### * ### * #
# ========================================================================== #


def kill_children():
    print(f'[END] Kill the kids!')
    import psutil
    child_processes = psutil.Process().children(recursive=True)
    for child in child_processes:
        print(f'[END] > Killing child process (PID={child.pid})')
        child.kill()


def set_seed(seed):
    if seed >= 0:
        print(f'[SEED] Setting seed to {seed}.')
    else:
        seed = random.randrange(2 ** 20)
        print(f'[SEED] Random seed not give, set to: {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # fixed input/model: ~5-10% speedup
    # torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    run_cli()