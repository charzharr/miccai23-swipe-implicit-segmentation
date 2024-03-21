

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class L2Norm(nn.Module):
    def __init__(self, weight=0.001):
        self.weight = weight 
        
    def forward(self, embedding):
        l2_penalty = (embedding ** 2).sum()
        l2_penalty *= self.weight 
        return l2_penalty


class SDFLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass
        
    def forward(self, pred, targ):
        pred = pred.tanh().squeeze(1)
        loss = F.l1_loss(pred, targ, reduction='none') #*1/(sdf_gt+1e-3)
        # loss = loss.mean(dim=1)
        loss = loss.mean() 
        
        return {
            'loss': loss,
        }
        # loss = None
        # loss = loss_i.sum(-1).mean() 
        #   # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
        
    def get_loss_string(self, out):
        loss = out['loss'].item()
        return f'loss {loss:.2f}'