""" Module metrics/classif_metrics.py (By: Charley Zhang, Jan 2022)

Some parts modified from:
https://github.com/Project-MONAI/MONAI/tree/dev/monai/metrics
"""

import torch
import torch.nn.functional as F
import numpy as np

from .unify import (reshape, stack, to_float, to_int, allclose, nan_to_num,
                    any as uni_any, sum as uni_sum)


# ========================================================================== #
# * ### * ### * ### *             Top-level API          * ### * ### * ### * #
# ========================================================================== #

def batch_metrics(preds, targs, num_classes):
    """
    Args:
        preds (tensor): BxC
        targs (tensor): BxC
    Returns: dictionary of overall and class-wise classification metrics like
        confusion matrix (and class-wise TP,FP,FN,TN), accuracy, f1, recall,
        precision, sensitivity, specificity.
    """
    def mean_with_nans(array):
        num_nans = np.count_nonzero(np.isnan(array))
        array = np.nan_to_num(array, nan=0)
        return array.sum() / (array.size - num_nans)
    
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu()
    if isinstance(targs, torch.Tensor):
        targs = targs.detach().cpu()
    
    # Sanity check (confusion matrix comparison with sklearn)
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    skcm = confusion_matrix(targs, preds, labels=np.arange(num_classes))
    sk_tp = np.array([skcm[i, i] for i in range(num_classes)])
    sk_fn = np.array([skcm[i].sum() - skcm[i, i] for i in range(num_classes)])
    sk_fp = np.array([skcm[:, i].sum() - skcm[i, i] for i in range(num_classes)])
    tns = []
    for c in range(num_classes):
        tn_mat = skcm.copy()
        tn_mat[c,:] = 0
        tn_mat[:, c] = 0
        tns.append(tn_mat.sum())
    sk_tn = np.array(tns)

    preds_1h = F.one_hot(preds.long(), num_classes=num_classes).to(torch.uint8)
    targs_1h = F.one_hot(targs.long(), num_classes=num_classes).to(torch.uint8)
    confusion_d = batch_confusion_matrix(preds_1h, targs_1h)
    
    tp, fp, fn = confusion_d['tp'], confusion_d['fp'], confusion_d['fn'] # Cx
    tn = confusion_d['tn']
    assert np.allclose(sk_tp, tp) and np.allclose(sk_fp, fp)
    assert np.allclose(sk_fn, fn) and np.allclose(sk_tn, tn)

    # Metrics Calculations
    # acc_class = (tp + tn) / (tp + fn + fp + tn)
    acc_mean = tp.sum() / preds.shape[0]  # jaccard |pred == targ| / N
    sk_acc = accuracy_score(targs, preds)
    # assert abs(acc_mean - sk_acc) < 1e-4, f'{acc_mean:.2f}, sk: {sk_acc:.2f}'

    f1_class = 2 * tp / (2 * tp + fp + fn)
    f1_mean = mean_with_nans(f1_class)
    sk_f1 = f1_score(targs, preds, average='macro')
    # assert abs(f1_mean - sk_f1) < 1e-4, f'{f1_mean:.2f}, sk: {sk_f1:.2f}'
    
    recall_class = tp / (tp + fn)
    recall_mean = mean_with_nans(recall_class)
    
    precision_class = tp / (tp + fp)
    precision_mean = mean_with_nans(precision_class)
    
    specificity_class = tn / (tn + fp)
    specificity_mean = mean_with_nans(specificity_class)

    return {
        'confusion_full': skcm,
        'confusion_class': confusion_d,  # Cx
        'tp': tp.sum().item(),
        'fp': fp.sum().item(),
        'fn': fn.sum().item(),
        'tn': tn.sum().item(),
        'tp_class': tp,
        'fp_class': fp,
        'fn_class': fn,
        'tn_class': tn,
        'accuracy': acc_mean,
        'f1_class': f1_class,  # shape=(C,)
        'f1': f1_mean,
        'recall_class': recall_class,
        'recall': recall_mean,
        'precision_class': precision_class,
        'precision': precision_mean,
        'specificity_class': specificity_class,
        'specificity': specificity_mean
    }


def batch_confusion_matrix(pred, targ, ignore_background=True):
    """ Multi-class classification confusion matrix values. 
    Args:
        pred: BxC one-hot tensor or array
        targ: BxC one-hot tensor or array
    Returns:
        BxCx4 (Batch x Classes x TP,FP,TN,FN
    """
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    assert pred.ndim == 2

    # Flatten pred & targ to B x C x S (S = all pixels for seg, 1 for classif)
    B, C = targ.shape[:2]
    
    tp = uni_sum(to_int((pred + targ) == 2), axis=[0])  # Cx
    tn = uni_sum(to_int((pred + targ) == 0), axis=[0])  # Cx
    
    p = uni_sum(targ, axis=[0])  # Cx, count of all positives
    n = B - p  # Cx, count of all negatives

    fn = p - tp
    fp = n - tn
    
    assert (tp + tn + fn + fp).sum() == B * C
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}  



