""" Module metrics3d.py (By: Charley Zhang, 2021.05)
Metrics for 3D segmentation. 

File contains 3 sections:
(1) Grouped metrics - func that return mutliple metrics (optimized execution)
(2) Individual metrics 
(3) Helper utilities for 3D+ tensor/array operations

Notes of Usage & Assumptions:
 - Assumes 3D input (X, Y, Z), 2D metrics should be in metrics2d.py
 - Assumes torch-like channel-first view BxCxHxWxD or CxHxWxD
 - Assumes no differentiability requirements.
 - Top-level func 'standard_metrics' should be sufficient for most uses.
   This function takes in tensors or np arrays. 
   
Resources: 
 - https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
"""


import torch
import numpy as np



def standard_metrics(pred, targ, get_hausdorff=False):
    """ Returns dictionary of most commonly used segmentation metrics.
    Parameters
        pred - prediction matrix BxCxHxWxD
        targ - target gt matrix  Bx1xHxWxD
    Return 
      - Dict: element x class-wise (BxC) TP, TN, FP, FN,
        Class-wise(Cx) dice ceofficient, jaccard (IOU), and Hausdorff.
    """
    pred, targ = standardize_dims(pred, targ)
    tp, fp, fn = confusion_matrix(pred, targ)  # BxC matrix
    
    # class-wise dice loss
    tp_c, fp_c, fn_c = tp.sum(0), fp.sum(0), fn.sum(0)
    dice_c = 2 * tp_c / (2 * tp_c + fp_c + fn_c + 10**-7)
    jaccard_c = tp_c / (tp_c + fp_c + fn_c + 10**-7)
    recall_c = tp_c / (tp_c + fn_c + 10**-7)
    
    hausd_c = hausdorff(pred, targ) if get_hausdorff else None
    
    return {
        'tp': tp,  # TPs as BxC matrix
        'fp': fp,
        'fn': fn,
        'tp_c': tp_c, # _c stands for per class (a size C vector) 
        'fp_c': fp_c,
        'fn_c': fn_c,
        'dice_c': dice_c,  
        'jaccard_c': jaccard_c,
        'hausdorff_c': hausd_c,
        'sensitivity_c': recall_c,
    }


### ======================================================================== ###
### * ### * ### * ### *      3D Segmentation Metrics     * ### * ### * ### * ###
### ======================================================================== ###


def confusion_matrix(pred, targ):
    """ Return counts for TPs, FPs, FN, TNs. 
    Parameters
        pred - prediction tensor or np array [Bx[Cx]]HxWxD
        targ - target tensor or np array  [Bx[Cx,1x]]HxWxD
    Return
        Dict of tp, fp, fn values per instance and class BxC.
          ex. tp is BxC array where with bth row & cth col is the # of
            true positives for example b in the batch and class c. 
    """
    pred, targ = standardize_dims(pred, targ)
    axes = tuple(range(2, pred.ndim))

    tp = pred * targ
    tp = sum_tensor(tp, axes, keepdim=False)
    # import IPython; IPython.emb ed(); 
    fp = pred * (1 - targ)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = (1 - pred) * targ
    fn = sum_tensor(fn, axes, keepdim=False)
    
    return tp, fp, fn


def hausdorff(pred, targ):
    pass


### ======================================================================== ###
### * ### * ### * ### *          Helper Utilities        * ### * ### * ### * ###
### ======================================================================== ###


def sum_tensor(inp, axes, keepdim=False):
    """ 
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    """
    axes = np.unique(tuple(axes)).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def np_scatter(slf, dim, index, src):
    """
    Taken from:
        https://stackoverflow.com/questions/46065873/how-to-do-scatter-and-gather-operations-in-numpy
    Writes all values from the Tensor src into new matrix at the indices 
    specified in the index Tensor.
    Example:
        slf[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        slf[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        slf[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    Paramters
        slf - Target matrix to which values will be placed
        dim - The axis along which to index
        index - The indices of elements to scatter
        src: The source element(s) to scatter
    Return
        slf - Matrix with changed values
    """
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    if slf.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= slf.ndim or dim < -slf.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Behavior in PyTorch's scatter where dim < 0 
        dim = slf.ndim + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    slf_xsection_shape = slf.shape[:dim] + slf.shape[dim + 1:]
    if idx_xsection_shape != slf_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output " + \
                         "should be the same size")
    if (index >= slf.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 "+ \
                         "and (slf.shape[dim] -1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return slc

    # We use index and dim parameters to create idx
    #  idx is in a form that can be used as a NumPy advanced index for 
    #  scattering of src param. in self
    # import IPython; IPython.embed(); 
    idx = [[*np.indices(tuple(idx_xsection_shape)).reshape(index.ndim - 1, -1),
            index[tuple(make_slice(index, dim, i))].reshape(1, -1)[0]] \
            for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + \
                             "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " +
                             str(dim) + ", all dimensions of index and src " + \
                             "should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), 
                                      np.prod(idx_xsection_shape)))
        slf[idx] = src[src_idx]

    else:
        slf[tuple(idx)] = src
    return slf

def scatter(matrix, dim, index, src):
    if isinstance(matrix, torch.Tensor):
        return matrix.scatter_(dim, index.to(torch.int64), src)
    else:
        return np_scatter(matrix, dim, index, src)
    
        

def unsqueeze(matrix, dim=0):
    """ Unsqueeze a tensor or ndarray on dim. """
    if isinstance(matrix, torch.Tensor):
        return matrix.unsqueeze(dim)
    else: 
        return np.expand_dims(matrix, dim)


def standardize_dims(pred, targ):
    """ Standardize pred and targ to be BxCxHxWxD & BxCxHxWxD, resp.
    Parameters
        pred - can be HxWxD, CxHxWxD, or BxCxHxWxD (assume C to be # classes)
        targ - can be HxWxD, CxHxWxD, or BxCxHxWxD (C = N_C or 1)
    """
    assert pred.ndim in (3, 4, 5), (f"Invalid prediction shape {pshape}. "
        f"Prediction must be HxWxD, CxHxWxD, or CxHxWxD")
    assert targ.ndim in (3, 4, 5), (f"Invalid target shape {tshape}. "
        f"Prediction must be HxWxD, CxHxWxD, or CxHxWxD")
    assert pred.shape[-3:] == targ.shape[-3:], (f"Pred, Targ size mismatch! "
        f"Pred: {pred.shape[-3:]}, Targ: {targ.shape[-3:]}")
    
    with torch.no_grad():  # improve performance in case inputs are tensors
        # standardize prediction matrix
        if pred.ndim == 3:   # HxWxD
            std_pred = unsqueeze(unsqueeze(pred))
        elif pred.ndim == 4: # CxHxWxD
            std_pred = unsqueeze(pred)
        else:
            std_pred = pred
        
        # standardize target matrix to BxCxHxWxD one-hot
        C = std_pred.shape[1]
        # print(std_pred.shape)
        if targ.ndim == 3:   # HxWxD
            std_targ = unsqueeze(unsqueeze(targ))
            return std_pred, std_targ
        
        if targ.ndim == 4:        
            if targ.shape[0] == C:  
                return std_pred, unsqueeze(targ)
            if targ.shape[0] == 1:
                targ = unsqueeze(targ)
            else:
                raise f"Targ neither matches pred channels nor has 1 channel."
        
        # ndim = 5
        if targ.shape[1] == std_pred.shape[1]:
            return std_pred, targ
        
        # make into 1-hot from labels of single target channel
        if isinstance(targ, torch.Tensor):
            std_targ = torch.zeros(std_pred.shape, device=targ.device)
        else:
            std_targ = np.zeros(std_pred.shape)
        
        return std_pred, scatter(std_targ, 1, targ, 1)


### ======================================================================== ###
### * ### * ### * ### *                Tests             * ### * ### * ### * ###
### ======================================================================== ###


if __name__ == '__main__':
    
    ### Standardize Dims ###
    C = 2  # num_channels
    for pshape, tshape in [[(1, 2, 3),(1, 2, 3)], [(2, 1, 2, 3),(2, 1, 2, 3)],
                [(2, 1, 2, 3),(1, 1, 2, 3)], [(3, 2, 1, 2, 3),(3, 2, 1, 2, 3)],
                [(3, 2, 1, 2, 3),(3, 1, 1, 2, 3)]]:
        print(f"Testing shapes: pred{pshape}, targ{tshape}..", end='')
        pred_tens, targ_tens = torch.randn(pshape), torch.randint(0, C, tshape)
        p, t = standardize_dims(pred_tens, targ_tens)
        assert p.shape == t.shape
        assert p.ndim == 5
        
        pred_np = pred_tens.cpu().numpy()
        targ_np = targ_tens.cpu().numpy()
        p_np, t_np = standardize_dims(pred_np, targ_np)
        assert p_np.shape == t_np.shape
        assert p_np.ndim == 5
        
        assert (p.cpu().numpy() == p_np).all()
        assert (t.cpu().numpy() == t_np).all()
        
        print('..done ✔')
    
    ### Standard Metrics ###
    print(f"\n----------\nTesting Standard Metrics")
    pred = torch.randint(0, 2, (3, 2, 1, 2, 3))
    targ = torch.randint(0, 2, (3, 2, 1, 2, 3))
    print('Prediction:\n', pred, '\nTarget:\n', targ)
    
    d = standard_metrics(pred, targ)
    
    rtp, rfp, rfn = d['tp'].sum(), d['fp'].sum(), d['fn'].sum()
    tp = ((pred == 1) & (targ == 1)).sum()
    fp = ((pred == 1) & (targ == 0)).sum()
    fn = ((pred == 0) & (targ == 1)).sum()
    assert tp == rtp, f"{tp} {rtp}"
    assert fp == rfp, f"{fp} {rfp}"
    assert fn == rfn, f"{fn} {rfn}"
    print(f"Correct ✔")
    
