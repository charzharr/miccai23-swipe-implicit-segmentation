""" Module utils/train/serialization.py (Author: Charley Zhang, July 2022)

Default functionality for saving pytorch training checkpoints.
"""

import os
import pathlib
import torch
# import wandb


def save_checkpoint(config, state_dict, tracker, epoch, code_d, trigger_metrics,
                    criterion=None, optimizer=None, ema_state_dict=None,
                    save_most_recent=False, remove_prev_checkpoints=True):
    """
    Default functionality that will save experiment checkpoints for:
        config, model state_dict (optional state_dict of ema model),
        exp. tracker, epoch, and source code. 
    """
    
    # Create serialization dictionary of checkpoint items to save
    save_d = {
        'state_dict': state_dict,
        'code_dict': code_d,
        'tracker': tracker,
        'config': config,
        'epoch': epoch,
    }
    if criterion is not None:
        save_d['criterion'] = criterion 
    if optimizer is not None:
        save_d['optimizer'] = optimizer 
    if ema_state_dict is not None:
        save_d['ema_state_dict'] = ema_state_dict
    
    # Save model as most recent or best performing? (end tag determination)
    end = 'last'
    for met, max_gud in trigger_metrics.items():
        if tracker.is_best(met, max_better=max_gud):
            subset = met.split('_')[0]
            met_name = '_'.join(met.split('_')[2:]) 
            score = float(tracker.metrics_d[met][-1])
            score_str = f'{int(score):d}' if score.is_integer() else f'{score:.3f}'
            end = f"best-{subset}-{met_name}-{score_str}"
            print(f"(save_checkpoint) {end}")
            break
    if end == 'last' and not save_most_recent:
        return

    # Create directories in exp/artifacts
    curr_path = pathlib.Path(__file__).parent.absolute()
    src_path = [curr_path.parents[i] for i in range(len(curr_path.parents)) 
                if curr_path.parents[i].name == 'src'][0]
    exp_path = src_path / 'experiments' / config.experiment.name
    exp_save_path = exp_path / 'artifacts' / config.experiment.id
    os.makedirs(exp_save_path, exist_ok=True)
    
    fn_start = f"{config['experiment']['id']}_{config.experiment.name}_"
    # if 'sweep' in config.experiment and config.experiment.sweep:
    #     fn_start += f'sweep-{config.experiment.sweep_id}_run-{wandb.run.id}'

    # Remove previous best-score checkpoints
    if remove_prev_checkpoints:
        rm_files = [f for f in os.listdir(exp_save_path) if f[-3:] == 'pth' \
                    and fn_start in f]
        if rm_files:
            match_str = end
            if end != 'last':
                match_str = f'best-{subset}-{met_name}'
            for f in rm_files:
                if match_str in f:
                    rm_file = os.path.join(exp_save_path, f)
                    print(f"Deleting file -x {f}")
                    os.remove(rm_file)
    
    filename = fn_start + f'ep{epoch}_' + end + '.pth'
    save_path = os.path.join(exp_save_path, filename)
    print(f"Saving model -> {filename}")
    torch.save(save_d, save_path)