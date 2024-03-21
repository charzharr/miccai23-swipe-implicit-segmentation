""" Module utils/train/distributed.py (By: Charley Zhang, Feb 2022)
Basic utilities for distributed training via PyTorch.
"""

import torch


def setup_dist(rank, world_size):
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    
    
def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training
    """
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size == 1:
        return
    torch.distributed.barrier()