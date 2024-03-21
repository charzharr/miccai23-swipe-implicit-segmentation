""" Module utils/statistics.py (Author: Charley Zhang, 2021)

Statistics tracking utilies for metrics with Weights-and-Biases integration.
"""

import collections
import numbers
import warnings
# import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from statistics import mean


LOWER_IS_BETTER = ['loss', 'fn', 'fp', 'hausdorff', 'distance']

class WandBTracker:
    """ All-in-one concise experiment tracking after each epoch completion.
        - Tracks all essential iteration and epoch metrics.
        - Automatically logs epoch stats to wandb.
        - If wand_settings not given, acts as a local tracker.
    Parameters
        wandb_settings = {'name': .., 'project': .., 'config':.., 'notes':.. }
        metric_names = ['list', 'of', 'metric', 'names']
    """

    def __init__(self, metric_names=[], wandb_settings={}):
        self.metrics_d = { k: [] for k in metric_names } if metric_names else {}
        assert not wandb_settings, f'Disabled for code release.'
        if wandb_settings:
            self.use_wandb = True
            if isinstance(wandb_settings, dict):
                self._setup_wandb(wandb_settings)   # Disabled for release
        else:
            self.use_wandb = False
        self.steps = 0
    
    def update(self, metrics_d, log=False, summarize={}, verbose=False,
               after_epoch=0, force_summarize=False, step=None):
        """
        Args:
            force_summarize (bool): should only be active in last step
        """
        
        summary_keys = []
        for k, item in metrics_d.items():
            if k not in self.metrics_d:
                if verbose:
                    print(f" (StatTracker) Adding stat: {k}, Val: {item}")
                self.metrics_d[k] = [item]
            else:
                self.metrics_d[k].append(item)
                if verbose:
                    print(f"(StatTracker) Adding {k}: {item}")
                    
            if self.use_wandb and summarize and k in summarize['triggers']:
                summary_keys.append(k)
                
        for k in summary_keys:
            max_better = False if sum([m.lower() in k for m in \
                LOWER_IS_BETTER]) else True
            is_best = self.is_best(k, max_better=max_better, 
                                   after_epoch=after_epoch)
            if force_summarize or is_best:
                trigger = k
                update_d = {}
                for kk in summarize['saves']:
                    if kk not in self.metrics_d:
                        print(f" (WandDBTracker) WARNING: summary met {kk} not valid.")
                        continue
                    
                    if force_summarize:
                        N = min(len(self.metrics_d[kk]), 10)
                        avg = mean(self.metrics_d[kk][-N:])
                        update_d[f';last{N}_{kk}'] = avg
                        update_d[f';last_{kk}'] = self.metrics_d[kk][-1]
                    
                    if is_best and kk == trigger:
                        name = f"*{kk}"
                        update_d[name] = self.metrics_d[kk][-1]
                    elif is_best and kk != trigger:
                        name = f":{kk}_trigger({trigger})"
                        update_d[name] = self.metrics_d[kk][-1]
                    
                if is_best:
                    update_d[f'*beststep_{k}'] = self.steps
                # if update_d:
                #     wandb.run.summary.update(update_d)
            
        # if self.use_wandb and sum([s in k for s in summarize]):
        #     max_better = False if sum([m.lower() in k for m in \
        #         LOWER_IS_BETTER]) else True
        #     if self.best(k, max_better=max_better) == item:
        #         stepnum = len(self.metrics_d[k]) - 1
        #         wandb.run.summary.update({
        #             'best_' + k: item, 'best_' + k + '_step': stepnum})
        # if log and self.use_wandb:
        #     if step is not None:
        #         wandb.log(metrics_d, step=step)
        #     else:
        #         wandb.log(metrics_d)
        if log:
            self.steps += 1
    
    def best(self, metric_name, max_better=True):
        if metric_name not in self.metrics_d:
            print(f" (WandBTracker) Given k({metric_name}) not valid.")
            return None
        metric_hist = self.metrics_d[metric_name]
        return max(metric_hist) if max_better else min(metric_hist)

    def is_best(
            self, 
            metric_name, 
            metric_val=None,
            topk=1,
            max_better=True, 
            after_epoch=0
            ):
        
        if metric_name not in self.metrics_d:
            msg = (f"(WandBTracker.is_best) Given k({metric_name}) not valid "
                   f"among given metrics: \n   {self.metrics_d.keys()}")
            warnings.warn(msg)
            return False
        
        metric_hist = self.metrics_d[metric_name]
        if self.steps <= after_epoch:
            return False
        
        metric_hist = metric_hist[after_epoch:]
        sorted_metric_hist = sorted(metric_hist, 
                                    reverse=True if max_better else False)
        topk = min(topk, len(sorted_metric_hist))
        
        topk_metric_hist = sorted_metric_hist[:topk]
        
        if metric_val is None:
            metric_val = metric_hist[-1]
        
        # best = max(metric_hist) if max_better else min(metric_hist)
        # return True if metric_hist[-1] == best else False
        
        if metric_val >= topk_metric_hist[-1]:
            return True 
        return False
        

    def best_by_metrics(self, metric_name, max_better=True, after_epoch=0):
        if metric_name not in self.metrics_d:
            print(f" (WandBTracker) Given k({metric_name}) not valid.")
            return None
        N_epochs = len(self.metrics_d[metric_name])
        met_ep = [(self.metrics_d[metric_name][i], i) for i in range(N_epochs)]
        reverse = True if max_better else False
        met_ep = sorted(met_ep, key=lambda x: x[0], reverse=reverse)
        
        best_ep = -1
        for m, e in met_ep:
            if e > after_epoch:
                best_ep = e
                break
        if best_ep == -1:
            print(f" (WandBTracker) no metrics fit the criteria given.")
            return None

        best_mets = {}
        best_mets['epoch'] = best_ep
        for k, mets in self.metrics_d.items():
            if len(mets) > best_ep:
                best_mets[k] = mets[best_ep]
        return best_mets

    def plot(self, keys=None):
        names, vals = [], []
        if keys:
            for k in keys:
                if k not in self.metrics_d:
                    continue
                names.append(k)
                vals.append(self.metrics_d[k])
        else:
            for k, v in self.metrics_d.items():
                if v and not isinstance(v[0], collections.Sequence):
                    names.append(k)
                    vals.append(vals)

        num_cols = 4
        num_rows = len(names) // num_cols + 1
        fig = plt.figure(figsize=(20, 5*num_rows))
        for i, name in enumerate(names):
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.set_title(name)
            ax.plot(vals[i])
        plt.show()

    # def resume(self, wandb_settings={}):
    #     """ Called when restarting after training session(s) has/have finished."""
    #     if not self.use_wandb or not hasattr(self, 'wandb_run_id'):
    #         raise ValueError(f"Prev run did not have a wandb session.")
    #     if wandb_settings:
    #         wandb.init(
    #             resume=self.wandb_run_id,
    #             project=wandb_settings['project'],
    #             name=wandb_settings['name'], 
    #             config=wandb_settings['config'],
    #             notes=None if 'notes' not in wandb_settings else wandb_settings['notes']
    #         )
    #     else:
    #         wandb.init(resume=self.wandb_run_id)
            
    def _setup_wandb(self, wandb_settings):
        print(" > Initializing Weights and Biases run..")
        cfg = wandb_settings['config']
        name = cfg['experiment']['id'] + '_' + cfg['experiment']['name']
        self.wandb_run_id = wandb.util.generate_id()
        print(f"\t- ID: {self.wandb_run_id}  - Name: {name}")
        wandb.init(
            id=self.wandb_run_id,
            resume='allow',
            project=wandb_settings['project'],
            name=wandb_settings['name'], 
            config=wandb_settings['config'],
            notes=None if 'notes' not in wandb_settings else wandb_settings['notes'],
            group=wandb_settings['group'] if 'group' in wandb_settings else None,
            job_type=wandb_settings['job_type'] if 'job_type' in wandb_settings else None
        )


class EpochMeters:
    """ Updates every iteration and keeps track of accumulated stats. """
    
    def __init__(self):
        self.accums = {}
        self.ns = {}

    def update(self, metrics_d, n=1):
        for k, item in metrics_d.items():
            if k not in self.accums:
                self.accums[k] = item
                self.ns[k] = n
                continue
            self.accums[k] += item
            self.ns[k] += n

    def avg(self, no_avg=[]):
        ret = {}
        for k, v in self.accums.items():
            if k in no_avg:
                ret[k] = v
            else:
                ret[k] = v/self.ns[k]
        return ret


class EpochMetrics(dict):
    """ Aggregates all the final tracked metrics for an epoch. """

    def __init__(self, *args, **kwargs):
        super(EpochMetrics, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def update(self, new_metrics, keys=None, new_key_start=''):
        """ Update epoch metrics to be printed or serialized later. 
        Args:
            new_metrics: dictionary of new metrics to be updated
            keys: list of keys to look for in new_mets to update
            new_key_start: beginning of key string to be stored
                e.g. 'train_', then a new met called 'F1' -> 'train_f1'
        """
        if not new_metrics:
            return
        for k, v in new_metrics.items():
            if keys and k not in keys:
                continue
            new_k = str(new_key_start) + k.lower()
            if new_k in self.keys():
                warnings.warn(f'(EpochMetrics) Key {new_k} overwritten.')
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                v = v.tolist()  # scalars return scalars here, not lists
            self[new_k] = v
    
    def print(self, pre='', ret_string=False):
        max_key_len = max([len(k) for k in self.keys()])
        string = str(pre)
        for k, v in self.items():
            if isinstance(v, numbers.Number) and float(v).is_integer():
                v_string = f'{int(v):d}'
            elif isinstance(v, float):
                v_string = f'{v:.4f}'
            elif isinstance(v, collections.Sequence):
                v_string = f'{np.array(v)}'
            else:
                v_string = str(v)
            string += ' ' + f'{k}'.ljust(max_key_len + 2) + f'{v_string}\n'
            # string += f'  {k: <21} {v_string}\n'

        if ret_string:
            return string
        print(string, flush=True)

# Basic Tests
if __name__ == '__main__':
    import sys

    metrics = {
        'train_loss': [.98, .76, .56, .43, .58],
        'test_loss': [.99, .77, .25, .44, .59],
        'overall_fp': [300, 250, 80, 100, 120],
        'overall_dice': [.98, .67, .68, .69, .69]
    }
    cfg = {'experiment': {'id': 'test0000', 'name': 'hello'}}
    tracker = WandBTracker(wandb_settings={
        'name': '0003_lr02', 'project': 'bibm2020', 'id': '0000', 'config': cfg
    })
    for epoch in range(len(metrics['train_loss'])):
        epmeter = EpochMeters()
        for it in range(1, 5):
            epmeter.update({
                'TPs': np.array([1,2,3]),
                'F1': 1.1,
                'F1s': np.array([.2,.4,.6])
            })
        tracker.update({k: v[epoch] for k, v in metrics.items()},
            log=True, summarize=['_loss', 'overall_dice'])
    import IPython; IPython.embed(); import sys; sys.exit(1)
