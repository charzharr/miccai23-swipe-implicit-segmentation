""" Module losses3d.py (By: Charley Zhang, 2021.05)
Contains loss functions specifically for 3D imaging tasks.

Assumes the following:
 - Prediction & target variables are tensors.
 - Prediction & target variables have the same shape (target = one-hot format)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


REDUCTIONS = ['mean', 'sum', 'class', 'none']


class BYOL3d:
    def __init__(self):
        print(f'💠 Criterion: BYOL3d initiated.')
        
    def __call__(self, pred, targ):
        """
        Args:
            pred: features outputted from projection head + prediction head
                  NxCxHxWxD
            targ: features outputted by EMA model + EMA projection head
                  NxCxHxWxD
        """
        msg = f'Pred & Targ shape mismatch: {pred.shape} {targ.shape}.'
        assert pred.shape == targ.shape, msg
        
        pred = pred.view(pred.shape[0], -1)
        targ = targ.view(targ.shape[0], -1)
        pred_norm = F.normalize(pred, dim=-1, p=2)
        targ_norm = F.normalize(targ, dim=-1, p=2)
        loss = (2 - 2 * (pred_norm * targ_norm).sum(dim=-1)).mean()
        return {
            'loss': loss
        }


class DiceLoss3d:
    """ Good for both 2D and 3D tensors. Assumes pred.shape == targ.shape. 
    Whether it is soft-dice or dice depends on whether predictions are 
     thresholded to be a binary mask (dice) or as probabilities (soft-dice).
    """        
    def __init__(self, smooth=1.):
        self.smooth = smooth
    
    def __call__(self, pred, targ):
        return dice_loss(pred, targ, s=self.smooth)


class CrossEntropyLoss3d:
    """ Different from torch CE in 2 ways: (1) assumes pred.shape == targ.shape, 
        which means targ must be one-hot, (2) does not apply softmax to pred.
    """
    def __init__(self, weights=None, reduction='mean'):
        self.weights = weights
        self.reduction = reduction
    
    def __call__(self, pred, targ):
        return cross_entropy_loss(pred, targ, weights=self.weights,
                                  reduction=self.reduction)


class DiceCrossEntropyLoss3d:
    def __init__(self, weights=None, alpha=0.5):
        self.weights = weights
        self.alpha = alpha

        name = type(self).__name__
        print(f'💠 {name} initiated with weights={self.weights}, \n'
              f'   alpha={self.alpha}.')
    
    def __call__(self, pred, targ):
        """
        Args:
            pred: BxCxDxHxW logits
            targ: BxCxDxHxW one hot binary mask
        """
        # Activation
        pred = pred.softmax(1)
        ce_loss = cross_entropy_loss(pred, targ, weights=self.weights,
                                     reduction='mean')
        dc_loss = dice_loss(pred, targ, ignore_background=False)
        # print('Mine', dc_loss, ce_loss)
        return dc_loss + ce_loss
        return self.alpha * dc_loss + (1 - self.alpha) * ce_loss
    

        

### ======================================================================== ###
### * ### * ### * ### *           Functional             * ### * ### * ### * ###
### ======================================================================== ###


def dice_loss(pred, targ, s=1, ignore_background=False, reduction='mean'):
    """ Dice loss. Assumes targ is in one-hot format. 
    Parameters
        pred - prediction image probabilities [any shape]
        targ - binary label image [any shape]
        s (float) - smoothing factor added to the numerator and denominator.
    """
    assert pred.shape == targ.shape 
    assert 0 <= pred.min() <= pred.max() <= 1, '"pred" must be probabilities'

    # Calculate soft dice
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ
    B, C = pred.shape[:2]

    pred_flat = pred.view(B, C, -1).float()
    targ_flat = targ.view(B, C, -1).float()
    intersec = (pred_flat * targ_flat).sum(-1)  # Shape BxC
    dice = 1 - (2 * intersec + s) / (pred_flat.sum(-1) + targ_flat.sum(-1) + s)
    # Dice is BxC, reduce to mean?
    if reduction == 'sum':
        return dice.sum()
    else:
        return dice.mean()


def cross_entropy_loss(pred, targ, weights=None, reduction='mean'):
    """ CE loss w/probabilties. Assumes targ is in one-hot format. 
    Parameters
        pred - prediction probabilities [BxCxHxWxD]
        targ - binary label image [BxCxHxWxD]
        s (float) - smoothing factor added to the numerator and denominator.
    """
    if pred.shape != targ.shape:
        assert targ.shape[0] == pred.shape[0]
        assert targ.shape[1:] == pred.shape[2:]
    assert reduction in REDUCTIONS, f'Reduction {reduction} is not valid!'
    
    targ_ind = to_single_index(targ, keepdims=False)  # BxHxWxD
    loss = F.nll_loss(torch.log(pred), targ_ind, weight=weights, 
                      reduction=reduction)
    return loss
    

### ======================================================================== ###
### * ### * ### * ### *       Rudimentary Testing        * ### * ### * ### * ###
### ======================================================================== ###


def to_one_hot(target, C=None):
    """ 
    Parameters
        tensor (torch.Tensor) - Input of shape Bx1xHxWxD
        C (int) - Number of classes
            Note: if C is none, max(unique_values) is used
    """
    assert target.ndim == 5, 'Expected 5 dimensional input (Bx1xHxWxD)'
    assert target.shape[1] == 1, 'Target shape needs to be Bx1xHxWxD'
    if C is None:
        C = target.max() + 1
    one_hot_shape = (target.shape[0], C, *target.shape[2:])
    one_hot = torch.zeros(one_hot_shape, device=target.device)
    one_hot.scatter_(1, target, 1)
    return one_hot

def to_single_index(target, keepdims=True):
    """
    Parameters
        target (tensor) - one-hot target BxCxHxWxD
        keepdims (bool) - if True, returns Bx1xHxWxD else BxHxWxD
    """
    assert target.ndim == 5, 'Expected 5 dimensional input (BxCxHxWxD)'
    targ_ind = target.view(target.shape[0], target.shape[1], -1).argmax(1)
    targ_ind = targ_ind.view(target.shape[0], *target.shape[2:])
    if keepdims:
        targ_ind = targ_ind.unsqueeze(1)
    return targ_ind
    
if __name__ == '__main__':
    
    ### Cross Entropy ###
    print(f"Testing CE ..", end='')
    C = 4
    pred = torch.randn(3, C, 1, 2, 3)
    targ = torch.randint(0, C, (3, 1, 1, 2, 3))
    assert torch.all(targ == to_single_index(to_one_hot(targ)))  
    
    torch_ce = nn.CrossEntropyLoss()
    torch_loss = torch_ce(pred, targ.squeeze(1))
    
    targ_1h = to_one_hot(targ)
    loss = cross_entropy_loss(pred.softmax(1), targ_1h, reduction='mean')
    assert abs(torch_loss - loss) < 10**-6, \
           f'Torch {torch_loss.item():.4f}, Ours {loss.item():.4f}.'
    print(f"good ✔")
    
    # Now weight weights
    print(f"Testing CE w/class-weights..", end='')
    pred = torch.randn(3, C, 1, 2, 3)
    targ = torch.randint(0, C, (3, 1, 1, 2, 3))
    weights = torch.randint(1, 3, (C,)).float()
    assert torch.all(targ == to_single_index(to_one_hot(targ)))  
    
    for reduction in ('none', 'sum', 'mean'):
        print(f' reduction {reduction}..', end='')
        torch_ce = nn.CrossEntropyLoss(weight=weights, reduction=reduction)
        torch_loss = torch_ce(pred, targ.squeeze(1))

        targ_1h = to_one_hot(targ)
        # import IPython; IPython.embed(); 
        loss = cross_entropy_loss(pred.softmax(1), targ_1h, reduction=reduction,
                                  weights=weights)
        
        assert torch.all((torch_loss - loss) < 10**-6), \
               f'Torch {torch_loss:.4f}, Ours {loss:.4f}.'
    print(f"good ✔")
    
    