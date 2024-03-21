
import os
import warnings
import copy
import pathlib
import yaml
import pprint
import numpy as np



def get_config(filename, merge_default=True, search_dir=''):
    r""" Merges specified config (or none) with default cfg file. 
    
    Input scenarios:
        - "filename" is a direct or relative path directly to the config file
        - "filename" is searched for in "search_dir".
    Note:
        If merge_default is True and search_dir is not empty, then a an attempt
        to fine a default config is made (file name has "default" in it), if
        multiple are found, then the alphabetical first is used. 
    """
    print(f" > Loading config ({filename})..", end='')
    
    # 1. Load config into "experiment_cfg"
    if not filename:
        warnings.warn(f"(configs.py) WARNING: empty filename given.")
        experiment_cfg = {}
    else:
        if os.path.isfile(filename):
            cfg_file = filename
        else:  # look for config in search_dir
            if not search_dir:
                msg = ('If you give just a name for config file, then you must '
                       'specify the directory to search in')
                raise ValueError(msg)
            cfg_file = os.path.join(search_dir, filename)
            assert os.path.isfile(cfg_file), f'{filename} not in ({search_dir})'
        with open(cfg_file, 'r') as f:
            experiment_cfg = yaml.safe_load(f)
    
    # 2. Merge with default_dict if it exists
    if merge_default:
        if not search_dir:
            msg = ('If you have "merge_default" set, then you must '
                    'specify "search_dir"')
            raise ValueError(msg)
        default_cfg = [f for f in os.listdir(search_dir) if 'default' in f]
        if not default_cfg:
            warnings.warn(f'No default config found in {search_dir}.')
        if len(default_cfg) > 1:
            warnings.warn(f'Multiple default configs found: {default_cfg}.')
        with open(os.path.join(search_dir, default_cfg[0]), 'r') as f:
            default_cfg = yaml.safe_load(f)
        experiment_cfg = merge(default_cfg, experiment_cfg)
    
    # 3. Apply hyperparamter sampling if cfg.experiment.hpsamp is set
    hpsamp = None
    if 'experiment' in experiment_cfg:
        if 'hpsamp' in experiment_cfg['experiment']:
            hpsamp = experiment_cfg['experiment']['hpsamp']
    if hpsamp:
        if os.path.isfile(hpsamp):
            hpsamp_file = hpsamp
        else:  # look for config in search_dir
            if not search_dir:
                msg = ('If you give just a name for config file, then you must '
                       'specify the directory to search in')
                raise ValueError(msg)
            hpsamp_file = os.path.join(search_dir, hpsamp)
            msg = f'HpSamp file "{hpsamp_file}" not in ({search_dir})'
            assert os.path.isfile(hpsamp_file), msg
        with open(hpsamp_file, 'r') as f:
            hpsamp_cfg = yaml.safe_load(f)
        experiment_cfg = get_sampled_config(hpsamp_cfg, experiment_cfg)
    print(f" done.")
    
    # Create namespace (dot notation accessible) and return
    experiment_cfg = create_namespace(experiment_cfg)
    return experiment_cfg


# ============= ##   Nested Dict Namespace (Dot Notation)  ## ============= #

def create_namespace(nested_dict):
    if not isinstance(nested_dict, dict):
        return nested_dict
    new_dict = {k: create_namespace(nested_dict[k]) for k in nested_dict}
    return DotDict(new_dict)

class DotDict(dict):
    """ Dictionary that allows dot notation access (nested not supported). """
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getattr__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            raise AttributeError(item)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    def dict(self):
        """ Return nested dict. """
        ret_d = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                ret_d[k] = v.dict()
            else:
                ret_d[k] = copy.deepcopy(v)
        
        return ret_d
        
        

# class NameSpace(dict):
#     """ Enables dot notation for one layer in a nested dict config. """
#     def __init__(self, *args, **kwargs):
#         super(NameSpace, self).__init__(*args, **kwargs)
#         self.__dict__ = self


# def create_namespace(nested_dict):
#     if not isinstance(nested_dict, dict):
#         return nested_dict
#     new_dict = {k: create_namespace(nested_dict[k]) for k in nested_dict}
#     return NameSpace(new_dict)


# ============= ##  Config Sampling for Hyperparemter Tuning ## ============= #

SAMPLE_INSTRS = {'randint', 'randreal', 'sample'} 


def get_sampled_config(hparam_cfg, experiment_cfg):
    print(f"(HParam Sampling) Sampling values for hyperparameters in:")
    pprint.pprint(hparam_cfg)
    return nested_sample(hparam_cfg, experiment_cfg)


def nested_sample(hparam_cfg, cfg):
    for k, v in hparam_cfg.items():
        if isinstance(v, dict):
            cfg[k] = nested_sample(hparam_cfg[k], cfg[k])
        else:
            # print(k, v)
            instr = v
            sampled = sample(v)
            cfg[k] = sampled
            if isinstance(sampled, float):
                sampled = f'{sampled:.6f}'
            print(f"{k} = {sampled} ∼{v}.")
    return cfg


def sample(instr):
    
    def convert(val):
        if val.isdigit():
            return int(val)
        try:
            float(val)
        except ValueError:
            return val
        return float(val)

    if isinstance(instr, str):
        parts = instr.split('(')
        cmd = parts[0]
        if cmd in SAMPLE_INSTRS:
            import re
            attrs = re.findall(r"[0-9a-z.]+", parts[1])
            attrs = [convert(a) for a in attrs]
            if cmd == 'randint':
                assert len(attrs) == 2
                sampled = np.random.randint(int(attrs[0]), int(attrs[1]) + 1)
                return sampled
            elif cmd == 'randreal':
                assert len(attrs) == 2
                rand = np.random.rand()
                diff = float(attrs[1]) - float(attrs[0])
                sampled = float(attrs[0]) + rand * diff
                # print(f"Sampled {sampled} from '{instr}'.")
                return sampled
            elif cmd == 'sample':
                N = len(attrs)
                idx = np.random.randint(0, N)
                sampled = attrs[idx]
                # print(f"Sampled {sampled} from '{instr}'.")
                return sampled
        else:
            raise ValueError(f"Command {cmd} is not valid!")
    print(f"(Hparam Sample) WARNING: Instruction({instr}) is not valid.")
    return instr        


def merge(default_d, experiment_d):
    experiment_d = dict(experiment_d)
    merged_d = dict(default_d)
    for k, v in experiment_d.items():
        if k not in merged_d:
            merged_d[k] = v
        else:
            if isinstance(v, dict) and isinstance(merged_d[k], dict):
                v = merge(merged_d[k], v)
            merged_d[k] = v
    return merged_d
