""" Module run_experiment.py (Author: Charley Zhang, Dec 2022)
Experiments for training implicit shape models for medical image segmentation.
"""

# --- Basic Imports --- #
import sys, os
import math, random
import pathlib
import time
import warnings
import itertools
import inspect
import multiprocessing
import numbers, string
import collections
from tqdm import tqdm
from pprint import pprint
from collections import namedtuple, OrderedDict

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import SimpleITK as sitk

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torchvision.transforms as T
import albumentations as A
from skimage.transform import resize as nd_resize
from scipy.ndimage import rotate as nd_rotate

# --- Project Code Imports --- #
import data, lib, experiments
from lib.utils import devices, timers, statistics
from lib.utils.io import output, files

# Default training components / utils that can easily be overrided below
from lib.utils.devices import mem, ram
from lib.utils.train.configs import create_namespace
from lib.utils.train.distributed import synchronize
from lib.utils.train.criterion import get_criterion 
from lib.utils.train.optimizers import get_optimizer
from lib.utils.train.schedulers import get_scheduler
from lib.utils.train.serialization import save_checkpoint

# --- 🔶🔶 Experiment-specific Custom Imports 🔶🔶 --- #
from .data_setup import get_data_components
from .model_setup import get_model_components

from lib.metrics.seg_metrics import batch_metrics as compute_seg_metrics

# --- Training Tools and Constants (all-cap) --- #
curr_dir = pathlib.Path(__file__).parent.absolute()
save_dir = curr_dir / 'artifacts'
watch = timers.StopWatch()

# 🔶🔶 Module-Saving Imports 🔶🔶
from lib.metrics import seg_metrics
source_code = {
    'run': inspect.getsource(inspect.getmodule(inspect.currentframe())),
    'metrics': inspect.getsource(inspect.getmodule(seg_metrics))
} 

# 🔶🔶 Debug Flags 
VISUALIZE = False


CM = namedtuple('CM', ('tp', 'fp', 'fn', 'tn'))


# ========================================================================== #
# * ### * ### * ### *               Training             * ### * ### * ### * #
# ========================================================================== #


def run(rank, config):
    """
    Main entry point for experiment that is called by run.py in the home of 
    the project (usually the src directory).
    Args:
        rank: world index for DDP, if not using distributed give rank=0
        config: dict & dot-accessible experiment configuration 
    """
    
    from train import set_seed
    set_seed(config['experiment']['seed'])

    # ------------------ ##  Experiment Setup  ## ------------------ #
    
    output.header_one('I. Training Components Setup', rank=rank)
    gpu_indices = config.experiment.gpu_idxs
    device = config.experiment.device
    debug_config = config.experiment.debug
    debug_mode = config.experiment.debug.mode
    
    # Setup Weights & Biases and Experiment Tracker
    tracker = setup_tracker(config, print_info=True)

    # Model Setup
    output.header_three('Model Setup', rank=rank)
    models_d = get_model_components(config)
    model = models_d['model'].to(device)
    if not config.experiment.distributed and len(gpu_indices) > 1:
        print(f'  * {len(gpu_indices)} GPUs, using nn.DataParallel.')
        model = torch.nn.DataParallel(model)
    
    # Data Pipeline
    output.header_three('Data Setup', rank=rank)
    data_d = get_data_components(config)
    data_d = setup_overfit_batch(config, data_d)
    
    df = data_d['df']
    train_df = data_d['train_df']
    train_set = data_d['train_set']
    train_loader = data_d['train_loader']
    val_df = data_d['val_df']
    val_set = data_d['val_set']
    test_df = data_d['test_df']
    test_set = data_d['test_set']
    
    # Traner Setup
    output.header_three('Trainer Setup', rank=rank)
    
    from .trainers.trainer2d_swipe import Trainer   
    trainer = Trainer(config, model, data_d, tracker)
    
    
    # ------------------ ##  Training Action  ## ------------------ #
    output.header_one('II. Training', rank=rank)
    
    tot_epochs = config.train.epochs - config.train.start_epoch
    global_iter = 0
    for epoch in range(config.train.start_epoch, config.train.epochs):
        watch.tic(name='epoch')
        
        # --- Train Epoch --- # 
        train_epmeter = trainer.train_epoch(epoch)

        # --- Epoch Metrics Collection/Tracking --- # 
        epoch_mets = statistics.EpochMetrics()   # epoch metrics collector
        epoch_mets.update(train_epmeter.avg(), 
                          config.serialize.train_metrics, 
                          'train_ep_')

        # Val / Test Epoch Metrics
        eval_every_n = debug_config.evaluate_every_n_epochs
        is_last_epoch = epoch == config.train.epochs - 1
        force_test_after = config.serialize.force_test_after
        if is_last_epoch or epoch % eval_every_n == eval_every_n - 1:  
            
            # Eval Validation Set
            if val_set is not None:          
                output.subsubsection('Validation Metrics', rank=rank)
                val_mets = trainer.infer(val_set, epoch, 
                                         name='validation',
                                         overlap_perc=config.test.overlap,
                                         save_predictions=False)
                epoch_mets.update(val_mets, 
                                  config.serialize.test_metrics, 
                                  'val_ep_')
                
                # See if validation score is the best
                full_key_metric, max_better = config.serialize.save_metric
                key_metric = '_'.join(full_key_metric.split('_')[2:])
                curr_score = val_mets[key_metric]
                is_best_val = tracker.is_best(full_key_metric, 
                                              metric_val=curr_score, 
                                              topk=3, 
                                              max_better=max_better)
                
                best_score = tracker.best(f'val_ep_{key_metric}', 
                                        max_better=True)
                if is_best_val:
                    print(f' 🥳 Best val {key_metric}!'
                        f' {curr_score:.4f} beats {best_score:.4f} 🥳\n'
                        f'Force-testing after epoch index '
                        f'{int(force_test_after * tot_epochs)}.')
            else:
                is_best_val = False
            
            # Eval Test Set
            force_test = True if epoch/tot_epochs > force_test_after else False
            force_test = force_test or debug_mode
            if is_best_val or force_test:
                output.subsubsection('Test Metrics', rank=rank)
                
                save_mode = debug_config.save and \
                            config.serialize.save_test_predictions
                save_curr_epoch = debug_mode or (epoch > eval_every_n and 
                    epoch % (tot_epochs // 6) in tuple(range(eval_every_n)))
                save_predictions = save_mode and save_curr_epoch 
                
                test_mets = trainer.infer(test_set, epoch, 
                                          name='test',
                                          overlap_perc=config.test.overlap,
                                          save_predictions=save_predictions)
                epoch_mets.update(test_mets, 
                                  config.serialize.test_metrics, 
                                  'test_ep_')
        else:
            print(f'\n    * Skipping Eval! Eval every {eval_every_n} epochs.')
        
        # Print final metrics for epoch
        epoch_mets.print(pre="\nEpoch Stats\n-----------\n")

        # Log to tracker and (opt) wandb
        force_sum = True if epoch == config.train.epochs - 1 else False
        save_model_after = config.serialize.save_model_after
        tracker.update(epoch_mets, 
                       log=True, 
                       summarize=config.serialize.summarize,
                       after_epoch=0, 
                       force_summarize=force_sum)
        
        save_model = config.serialize.save_model and debug_config.save
        if not save_model:
            print(f'🚨  Model-saving functionality is off! \n')
        if save_model and (epoch >= save_model_after*tot_epochs or debug_mode):
            save_metric_l = config.serialize.save_metric
            save_checkpoint(config, 
                            model.state_dict(), 
                            tracker, 
                            epoch, 
                            source_code, 
                            {save_metric_l[0]: save_metric_l[1]},
                            save_most_recent=config.serialize.save_recent_model)
        
        # --- Final End-of-Epoch Housekeeping & Maintenance --- # 
        watch.toc(name='epoch')
        
        # 🔁 End of Epoch 🔁 #

    return tracker



# ========================================================================== #
# * ### * ### * ### *               Helpers              * ### * ### * ### * #
# ========================================================================== #


def setup_tracker(config, print_info=True):
    # Setup wandb and tracker
    wandb_settings = {}
    tracker = statistics.WandBTracker(wandb_settings=wandb_settings)
    
    # Print Basic Info
    if print_info:
        gpu_indices = config.experiment.gpu_idxs
        device = config.experiment.device
        use_amp = config.experiment.amp and bool(gpu_indices)
    
        print(f"[Experiment Settings (@supervised/emain.py)]", flush=True)
        print(f' > Amp: {use_amp}, Device: {device}, GPU-Idxs: {gpu_indices}')
        print(f" > Prepping train config..")
        print(f"\t - experiment:  {config.experiment.project} - "
                f"{config.experiment.name}, id({config.experiment.id})")
        print(f"\t - batch_size {config.train.batch_size}, "
                f"\t - start epoch: {config.train.start_epoch}/"
                f"{config.train.epochs},")
        print(f"\t - Optimizer ({config.train.optimizer.name}): "
                f"\t - lr {config.train.optimizer.lr}, "
                f"\t - wt_decay {config.train.optimizer.wt_decay} ")
        print(f"\t - Scheduler ({config.train.scheduler.name}): "
                f"\t - rampup: {config.train.scheduler.rampup_rates}\n")
    
    return tracker


def setup_overfit_batch(config, data_d):
    """ Overfits a few mini-batches for debugging purposes. """
    if not config.experiment.debug['overfitbatch']:
        return data_d 
    
    print(f'🚨  Overfitting a set of minibatches! \n')
    
    train_loader = data_d['train_loader']
    batches = []
    for i, batch in enumerate(train_loader):
        Y_id = batch['masks']
        vol = Y_id.shape[2] * Y_id.shape[3] * Y_id.shape[4]
        ids, fg_counts = Y_id.unique(return_counts=True)
        fg_vol = fg_counts[1:].sum()
        if len(ids > 4) and fg_vol > 0.25 * vol:
            print(f'Batch {i+1} successfully meets the criteria '
                    f'(volume: {fg_vol / vol}).')
            batches.append(batch)
        if len(batches) >= 2: 
            break
    train_loader = list(itertools.islice(itertools.cycle(batches), 50))
    data_d['train_loader'] = train_loader
    
    val_set = data_d['val_set']
    if val_set is not None:
        val_set._samples = [val_set.samples[0]]
        data_d['val_set'] = train_loader
        
    test_set = data_d['test_set']
    if test_set is not None:
        test_set._samples = [test_set.samples[0]]
        data_d['test_set'] = train_loader
        
    return data_d


