""" lib/metrics/seg_metrics.py  (Author: Charley Zhang, 2021)
Most commonly used segmentation metrics (or the ones I need for med-im projects.
Huge help from MONAI: https://www.github.com/Project-MONAI/MONAI

Usage Philosophy:
    - Can take either detached cpu tensors or numpy arrays
    - Returns all information (per example & per class results). It is up to
        functions lower in the stack to aggregate in the form the application
        requires. 
            e.g. batch_confusion_matrix returns a BxCx4

List of Metrics:
    - Confusion Matrix (2D & 3D, Classif & Seg)
    - Dice (2D & 3D, Seg)
    - Jaccard (2D & 3D, Seg)
    - Hausdorff (2D & 3D, Seg)
"""

from genericpath import exists
import warnings
from collections import namedtuple

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance

from lib.utils.dotdict import DotDict
# from dmt.metrics.medpy_metrics import dc, jc, hd 
from .unify import (reshape, stack, to_float, to_int, allclose, nan_to_num,
                    any as uni_any, sum as uni_sum)
from .seg_utils import get_mask_edges, get_surface_distance


__all__ = ['batch_metrics', 
           'batch_cdj_metrics',
           'batch_confusion_matrix', 
           'batch_dice',
           'batch_jaccard',
           'batch_hausdorff']


# Module scope for multiprocessing compatibility
SegCM = namedtuple('SegCM', ('tp', 'fp', 'fn', 'tn'))
Mets = namedtuple('Mets', ('confusion', 'dice', 'jaccard', 'exists'))


def batch_metrics(preds, targs, ignore_background=False, naive_avg=True,
                  hd=False, hd_percentile=95):
    """ Compute dice, jaccard, and confusion info. 
        Principle: each entry is itself an endpoint metric. 
    Args:
        preds (tensor or array): BxC(xD)xHxW (binary one-hot)
        targs: (tensor or array) BxC(xD)xHxW (binary one-hot)
        naive_avg: just set to False for inference. May skew results if 
            a given pred & targ batch doesn't have all classes.
    """
    assert preds.shape == targs.shape
    # assert preds.ndim == 5
    assert 'int' in str(preds.dtype)
    assert 'int' in str(targs.dtype)
    
    def mean_with_nans(array):
        num_nans = np.count_nonzero(np.isnan(array))
        array = np.nan_to_num(array, nan=0)
        return array.sum() / (array.size - num_nans)
    
    cdj_ntuple = batch_cdj_metrics(preds, targs,
                                  ignore_background=ignore_background)

    # cm = conf-matrix. Named_tuple: tp, fp, tn, fn (each BxC)
    bc_cm = SegCM(np.array(cdj_ntuple.confusion.tp), 
                  np.array(cdj_ntuple.confusion.fp), 
                  np.array(cdj_ntuple.confusion.fn), 
                  np.array(cdj_ntuple.confusion.tn))
    bc_tp, bc_fn, bc_fp = bc_cm.tp, bc_cm.fn, bc_cm.fp
    
    bc_dice = np.array(cdj_ntuple.dice)  # BxC, bc = batch x class dice
    bc_jaccard = np.array(cdj_ntuple.jaccard) # BxC
    bc_exists = np.array(cdj_ntuple.exists) # BxC (1 if class exists, else 0)
    
    if hd:
        bc_hd = batch_hausdorff(preds, targs, 
                                ignore_background=ignore_background,
                                percentile=hd_percentile)
    
    # with np.errstate(divide='ignore', invalid='ignore'):
    # exists_class = to_float((bc_tp.sum(0) + bc_fn.sum(0) > 0))  # xC
    # exists_batch = to_float((bc_tp.sum(1) + bc_fn.sum(1) > 0))  # xB
    
    class_counts = bc_exists.sum(0)
    batch_counts = bc_exists.sum(1)
    B, C = bc_dice.shape[0], bc_dice.shape[1]
    
    if naive_avg:
        tps, fps, fns = bc_tp.sum(), bc_fp.sum(), bc_fn.sum()
        
        dice_class = bc_dice.mean(0)
        dice_class_agg = (2 * bc_tp.sum(0)) / (2 * bc_tp.sum(0) + \
                          bc_fp.sum(0) + bc_fn.sum(0) + 1e-8)
        dice_batch = bc_dice.mean(1)
        dice_batch_agg = (2 * bc_tp.sum(1)) / (2 * bc_tp.sum(1) + \
                          bc_fp.sum(1) + bc_fn.sum(1) + 1e-8)
        dice_summary = dice_batch.mean()
        dice_summary_agg = (2 * tps) / (2 * tps + fps + fns + 1e-8)
        
        jaccard_class = bc_jaccard.mean(0)
        jaccard_class_agg = bc_tp.sum(0) / (bc_tp.sum(0) + \
                            bc_fp.sum(0) + bc_fn.sum(0) + 1e-8)
        jaccard_batch = bc_jaccard.mean(1)
        jaccard_batch_agg = bc_tp.sum(1) / (bc_tp.sum(1) + \
                            bc_fp.sum(1) + bc_fn.sum(1) + 1e-8)
        jaccard_summary = jaccard_batch.mean()
        jaccard_summary_agg = tps / (tps + fps + fns + 1e-8)
    else:   
        def sum(values, counts):
            """ Here, counts is an indicator where a value is added if >0 """
            ret = 0
            for i, (v, c) in enumerate(zip(values.flatten(), counts.flatten())):
                if c == 0:
                    continue
                ret += v
            return ret
        
        # 1. Recompute bc_dice & bc_jaccard from bc_cm
        adj_bc_dice = np.zeros_like(bc_dice)
        adj_bc_jaccard = np.zeros_like(bc_jaccard)        
        for b in range(B):
            for c in range(C):
                if bc_exists[b][c]:
                    tp, fp, fn = bc_tp[b][c], bc_fp[b][c], bc_fn[b][c]
                    adj_bc_dice[b][c] = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                    adj_bc_jaccard[b][c] = tp / (tp + fp + fn + 1e-8)
                else:
                    adj_bc_dice[b][c] = np.NaN 
                    adj_bc_jaccard[b][c] = np.NaN
        bc_dice, bc_jaccard = adj_bc_dice, adj_bc_jaccard  # nonexist to nans
        
        dice_class, dice_class_agg = np.zeros((C)), np.zeros((C))
        jaccard_class, jaccard_class_agg = np.zeros((C)), np.zeros((C))
        for c in range(C):
            N = (bc_exists[:, c] > 0).sum()
            if N == 0:
                dice_class[c] = np.NaN
                dice_class_agg[c] = np.NaN
                jaccard_class[c] = np.NaN
                jaccard_class_agg[c] = np.NaN
                continue
            dice_class[c] = sum(adj_bc_dice[:,c], bc_exists[:,c]) / N
            jaccard_class[c] = sum(adj_bc_jaccard[:,c], bc_exists[:,c]) / N
            dice_class_agg[c] = (2 * sum(bc_tp[:,c], bc_exists[:,c])) / (
                                 2 * sum(bc_tp[:,c], bc_exists[:,c]) + 
                                 sum(bc_fp[:,c], bc_exists[:,c]) + 
                                 sum(bc_fn[:,c], bc_exists[:,c]) + 1e-8)
            jaccard_class_agg[c] = sum(bc_tp[:,c], bc_exists[:,c]) / (
                                   sum(bc_tp[:,c], bc_exists[:,c]) + 
                                   sum(bc_fp[:,c], bc_exists[:,c]) + 
                                   sum(bc_fn[:,c], bc_exists[:,c]) + 1e-8)
        
        dice_batch, dice_batch_agg = np.zeros((B)), np.zeros((B))
        jaccard_batch, jaccard_batch_agg = np.zeros((B)), np.zeros((B))
        for b in range(B):
            N = (bc_exists[b, :] > 0).sum()
            if N == 0:
                dice_batch[b] = np.NaN
                jaccard_batch[b] = np.NaN
                dice_batch_agg[b] = np.NaN 
                jaccard_batch_agg[b] = np.NaN 
                continue
            dice_batch[b] = sum(adj_bc_dice[b, :], bc_exists[b, :]) / N 
            jaccard_batch[b] = sum(adj_bc_jaccard[b,:], bc_exists[b,:]) / N
            dice_batch_agg[b] = (2 * sum(bc_tp[b,:], bc_exists[b,:])) / (
                                 2 * sum(bc_tp[b,:], bc_exists[b,:]) + 
                                 sum(bc_fp[b,:], bc_exists[b,:]) + 
                                 sum(bc_fn[b,:], bc_exists[b,:]) + 1e-8)
            jaccard_batch_agg[b] = sum(bc_tp[b,:], bc_exists[b,:]) / (
                                   sum(bc_tp[b,:], bc_exists[b,:]) + 
                                   sum(bc_fp[b,:], bc_exists[b,:]) + 
                                   sum(bc_fn[b,:], bc_exists[b,:]) + 1e-8)
        # OLD: dice_batch = average(bc_dice.sum(1), batch_counts)  # WRONG!
        
        tps = sum(bc_tp, bc_exists)
        fps = sum(bc_fp, bc_exists)
        fns = sum(bc_fn, bc_exists)
        dice_summary_agg = (2 * tps) / (2 * tps + fps + fns + 1e-8)
        jaccard_summary_agg = tps / (tps + fps + fns + 1e-8)
        
        denom = np.count_nonzero(~np.isnan(dice_batch)) + 1e-8
        dice_summary = np.nansum(dice_batch) / denom
        
        denom = np.count_nonzero(~np.isnan(jaccard_batch)) + 1e-8
        jaccard_summary = np.nansum(jaccard_batch) / denom
        
    ret_d = DotDict({
        'confusion_all': bc_cm,                     # named tuple (tp,fp,fn,tn BxC)
        'tps': tps,
        'fps': fps,
        'fns': fns,
        'recall_agg': tps / (tps + fns + 1e-8), 
        'precision_agg': tps / (tps + fps + 1e-8), 
        'exists_all': bc_exists,                    # BxC (1 or 0)
        'exists_class': class_counts,               # Cx (int >= 0)
        'exists_batch': batch_counts,               # Bx (int >= 0)
        'dice_all': bc_dice,                        # BxC
        'dice_class': dice_class,                   # Cx (average across inst)
        'dice_class_agg': dice_class_agg,           # Cx (aggregated TPs, etc)
        'dice_batch': dice_batch,                   # Bx (average across class)
        'dice_batch_agg': dice_batch_agg,           # Bx 
        'dice_summary': dice_summary,               # Float (avgerage across inst)
        'dice_summary_agg': dice_summary_agg,       # Float (aggregated TPs)
        'jaccard_all': bc_jaccard,                  # BxC
        'jaccard_class': jaccard_class,             # Cx (average across inst)
        'jaccard_class_agg': jaccard_class_agg,     # Cx (aggregated TPs, etc)
        'jaccard_batch': jaccard_batch,             # Bx (average across class)
        'jaccard_batch_agg': jaccard_batch_agg,     # Bx 
        'jaccard_summary': jaccard_summary,         # Float (avgerage across inst)
        'jaccard_summary_agg': jaccard_summary_agg, # Float (aggregated TPs)
    })
    
    if hd:
        nans_per_image = np.isnan(bc_hd).sum(1)  # Bx
        denom = bc_hd.shape[1] - nans_per_image
        b_hd = np.zeros(B)
        for b in range(B):
            if denom[b] == 0:
                b_hd[b] = np.nan
            else:
                b_hd[b] = np.nan_to_num(bc_hd[b], nan=0).sum() / denom[b]
        
        hd_summary = mean_with_nans(b_hd)
        ret_d['hd_summary'] = hd_summary
    
    return ret_d


def batch_cdj_metrics(pred, targ, ignore_background=True):
    """ Optimized execution to get Confusion Matrix, Dice, and Jaccard.
        Called by batch_metrics. 
    Args:
        pred: BxC(xD)xHxW tensor or array
        targ: BxC(xD)xHxW tensor or array
        ignore_background (bool): flag to ignore first channel dim or not
    Returns:
        A namedtuple that has the following keys:
            - 'confusion' a namedtuple that has tp, fp, fn, tn arrays
            - 'jaccard' a BxC array of ious (nans are turned into 0s)
            - 'dice' a BxC array of dice scores (nans are turned into 0s)
    """
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
    CM = batch_confusion_matrix(pred, targ, ignore_background=ignore_background)
    tp = CM.tp  # BxC
    fp = CM.fp
    fn = CM.fn
    exists = to_float((tp + fn) > 0)  # if b, c has item, then 1
    
    # Get Dice & Jaccard
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)  # BxC
    # dice = nan_to_num(dice, nan=0)
    # dice_sanity = batch_dice(pred, targ)
    # assert allclose(dice, dice_sanity)

    jaccard = tp / (tp + fp + fn + 1e-8)  # BxC
    # jaccard = nan_to_num(jaccard, nan=0)
    # jaccard_sanity = batch_jaccard(pred, targ)
    # assert allclose(jaccard, jaccard_sanity)

    return Mets(CM, dice, jaccard, exists)


# ------------ ##  Individual Metrics: CM, Dice, Jaccard  ## ----------- # 

def batch_confusion_matrix(pred, targ, ignore_background=True):
    """ 2D or 3D image for segmenation. 
    Args:
        pred: BxC(xD)xHxW tensor or array
        targ: BxC(xD)xHxW tensor or array
        ignore_background (bool): flag to ignore first channel dim or not
    Returns:
        BxCx4 (Batch x Classes x TP,FP,TN,FN
    """
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
#     if isinstance(pred, torch.Tensor):
#         pred = pred.float()
#         targ = targ.float()
    
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ

    # Flatten pred & targ to B x C x S (S = all pixels for seg, 1 for classif)
    B, C = targ.shape[:2]
    pred_flat = reshape(pred, (B, C, -1))
    targ_flat = reshape(targ, (B, C, -1))
    
    # tp = to_int((pred_flat + targ_flat) == 2)  # BxCxS
    # tn = to_int((pred_flat + targ_flat) == 0)  # BxCxS
    tp = (pred_flat + targ_flat) == 2   # BxCxS
    tn = (pred_flat + targ_flat) == 0   # BxCxS

    tp = uni_sum(tp, axis=[2])  # BxC
    tn = uni_sum(tn, axis=[2])  # BxC
    
    p = uni_sum(targ_flat, axis=[2])  # BxC, count of all positives
    n = pred_flat.shape[-1] - p  # BxC, count of all negatives

    fn = p - tp
    fp = n - tn

    return SegCM(tp, fp, fn, tn)  # can be int or long if inputs are int


def batch_dice(pred, targ, ignore_background=True):
    """
    Args:
        pred: BxCxDxHxW binary array
        targ: BxCxDxHxW binary array
    """    
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ
    
    B, C = targ.shape[0], targ.shape[1]
    if isinstance(pred, np.ndarray):
        elem_class_dice = np.zeros((B, C)).astype(np.float32)
    else:
        elem_class_dice = torch.zeros((B, C), dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            elem_class_pred = pred[b, c]
            elem_class_targ = targ[b, c]
            
            intersection = np.count_nonzero(elem_class_pred & elem_class_targ)
            pred_area = np.count_nonzero(elem_class_pred)
            targ_area = np.count_nonzero(elem_class_targ)

            denom = pred_area + targ_area
            if denom == 0:
                elem_class_dice[b, c] = 0.
            else:
                elem_class_dice[b, c] = 2. * intersection / denom
    
    return elem_class_dice


def batch_jaccard(pred, targ, ignore_background=True):
    """
    Args:
        pred: BxCxDxHxW binary array
        targ: BxCxDxHxW binary array
    """    
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ
    
    B, C = targ.shape[0], targ.shape[1]
    if isinstance(pred, np.ndarray):
        elem_class_jaccard = np.zeros((B, C)).astype(np.float32)
    else:
        elem_class_jaccard = torch.zeros((B, C), dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            elem_class_pred = pred[b, c]
            elem_class_targ = targ[b, c]
            
            intersection = np.count_nonzero(elem_class_pred & elem_class_targ)
            pred_area = np.count_nonzero(elem_class_pred)
            targ_area = np.count_nonzero(elem_class_targ)
            union = pred_area + targ_area - intersection
            
            if union <= 0:
                elem_class_jaccard[b, c] = 0.
            else:
                elem_class_jaccard[b, c] = intersection / union
    
    return elem_class_jaccard


# ------------ ##  Hausdorff Metric & Helpers  ## ----------- # 

def batch_hausdorff(pred, targ, ignore_background=True, percentile=None):
    """ 2D or 3D image for segmenation. 
    Args:
        pred: BxC(xD)xHxW tensor or array
        targ: BxC(xD)xHxW tensor or array
        ignore_background (bool): flag to ignore first channel dim or not
        percentile (float): between 0 and 100
    Returns:
        BxC (Batch x Classes) scores. 
    """
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'

    hd = compute_hausdorff_distance(pred, targ, 
                                    include_background=not ignore_background,
                                    percentile=percentile)
    return hd
    
# def batch_hausdorff(
#         pred, 
#         targ, 
#         ignore_background=True,
#         distance_metric='euclidean',
#         percentile=None,
#         directed=False
#         ):
#     """ 
#     Args: 
#         pred: BxCxDxHxW binary array or tensor
#         targ: BxCxDxHxW binary array or tensor
#         ignore_background: flag to take out 1st class dimension or not
#         distance_metric: 'euclidean', 'chessboard', 'taxicab'
#         percentile: [0, 100], return percentile of distance rather than max.
#         directed: flag to calculated directed Hausdorff distance or not.
#     """
#     assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
#     assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
#     assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
#     if ignore_background:
#         pred = pred[:, 1:] if pred.shape[1] > 1 else pred
#         targ = targ[:, 1:] if targ.shape[1] > 1 else targ

#     B, C = targ.shape[:2]
#     if isinstance(pred, np.ndarray):
#         HD = np.zeros((B, C)).astype(np.float32)
#     else:
#         HD = torch.zeros((B, C), dtype=torch.float32)
    
#     for b, c in np.ndindex(B, C):
#         (edges_pred, edges_gt) = get_mask_edges(pred[b, c], targ[b, c])
#         if not uni_any(edges_gt):
#             warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan/inf distance.")
#         if not uni_any(edges_pred):
#             warnings.warn(f"the prediction of class {c} is all 0, this may result in nan/inf distance.")

#         distance_1 = compute_percent_hausdorff_distance(edges_pred, edges_gt, distance_metric, percentile)
#         if directed:
#             HD[b, c] = distance_1
#         else:
#             distance_2 = compute_percent_hausdorff_distance(edges_gt, edges_pred, distance_metric, percentile)
#             HD[b, c] = max(distance_1, distance_2)
#     return torch.from_numpy(hd)




# # ------------ ##  Rudimentary Tests  ## ----------- # 

# if __name__ == '__main__':
    
#     pred_array = np.random.randint(0, 2, size=(2, 3, 5, 5))
#     targ_array = np.random.randint(0, 2, size=(2, 3, 5, 5))
    
#     pred_tens = torch.tensor(pred_array)
#     targ_tens = torch.tensor(targ_array)
    
#     # Confusion Matrix Tests
#     cm_tens = batch_confusion_matrix(pred_tens, targ_tens)
#     cm_array = batch_confusion_matrix(pred_array, targ_array)
    
#     assert allclose(cm_tens.numpy(), cm_array)
    
#     # CM Package
#     cdj_tens = batch_cdj_metrics(pred_tens, targ_tens)
#     cdj_array = batch_cdj_metrics(pred_array, targ_array)
#     assert allclose(cdj_tens.confusion.numpy(), cdj_array.confusion)
#     assert allclose(cdj_tens.dice.numpy(), cdj_array.dice)
#     assert allclose(cdj_tens.jaccard.numpy(), cdj_array.jaccard)
    