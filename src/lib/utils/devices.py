""" Module utils/devices.py (Author: Charley Zhang, 2022)

Get device or hardware usage information.
"""

import os, psutil
import torch
import subprocess
import collections


def mem(gpu_indices):
    """ Get primary GPU card memory usage. """
    
    if not torch.cuda.is_available():
        return -1.
    mem_map = get_gpu_memory_map()
    
    if isinstance(gpu_indices, collections.abc.Sequence):
        prim_card_num = gpu_indices[0]
    else:
        prim_card_num = int(gpu_indices)
    return mem_map[prim_card_num] / 1000


def ram(disp=False):
    """ Return (opt display) RAM usage of current process in megabytes. """
    
    process = psutil.Process(os.getpid())
    bytes = process.memory_info().rss
    mbytes = bytes // 1048576
    sys_mbytes = psutil.virtual_memory().total // 1048576
    if disp:
        print(f'🖥️  Current process (id={os.getpid()}) '
              f'RAM Usage: {mbytes:,} MBs / {sys_mbytes:,} Total MBs.')
    return mbytes


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    
    return gpu_memory_map
