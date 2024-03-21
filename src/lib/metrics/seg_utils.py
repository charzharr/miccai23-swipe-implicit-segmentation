
import itertools
import torch 
import numpy as np

from scipy.ndimage.morphology import (binary_erosion, distance_transform_edt,
                                      distance_transform_cdt)

from .unify import reshape, stack, sum, to_float, allclose, any



# ------------ ##  Utilities for Surface & Mask Analysis  ## ----------- # 

def get_mask_edges(seg_pred, seg_gt, label_idx=1, crop=True):
    """ (from MONAI)
    Do binary erosion and use XOR for input to get the edges. This
    function is helpful to further calculate metrics such as Average Surface
    Distance and Hausdorff Distance.
    The input images can be binary or labelfield images. If labelfield images
    are supplied, they are converted to binary images using `label_idx`.
    `scipy`'s binary erosion is used to to calculate the edges of the binary
    labelfield.
    In order to improve the computing efficiency, before getting the edges,
    the images can be cropped and only keep the foreground if not specifies
    ``crop = False``.
    We require that images are the same size, and assume that they occupy the
    same space (spacing, orientation, etc.).
    Args:
        seg_pred: the predicted binary or labelfield image.
        seg_gt: the actual binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        crop: crop input images and only keep the foregrounds. In order to
            maintain two inputs' shapes, here the bounding box is achieved
            by ``(seg_pred | seg_gt)`` which represents the union set of two
            images. Defaults to ``True``.
    """

    # Get both labelfields as np arrays
    if isinstance(seg_pred, torch.Tensor):
        seg_pred = seg_pred.detach().cpu().numpy()
    if isinstance(seg_gt, torch.Tensor):
        seg_gt = seg_gt.detach().cpu().numpy()

    if seg_pred.shape != seg_gt.shape:
        raise ValueError("seg_pred and seg_gt should have same shapes.")

    # If not binary images, convert them
    if seg_pred.dtype != bool:
        seg_pred = seg_pred == label_idx
    if seg_gt.dtype != bool:
        seg_gt = seg_gt == label_idx

    if crop:
        if not np.any(seg_pred | seg_gt):
            return (np.zeros_like(seg_pred), np.zeros_like(seg_gt))

        seg_pred, seg_gt = np.expand_dims(seg_pred, 0), np.expand_dims(seg_gt, 0)
        box_start, box_end = generate_spatial_bounding_box(
                                np.asarray(seg_pred | seg_gt))
        # cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        # seg_pred, seg_gt = np.squeeze(cropper(seg_pred)), np.squeeze(cropper(seg_gt))
        import IPython; IPython.embed(); 
        slices = [slice(None)] + [slice(s, e) for s, e in zip(box_start, box_end)]
        seg_pred_crop = seg_pred[tuple(slices)]
        seg_gt_crop = seg_gt[tuple(slices)]
        

    # Do binary erosion and use XOR to get edges
    edges_pred = binary_erosion(seg_pred) ^ seg_pred
    edges_gt = binary_erosion(seg_gt) ^ seg_gt

    return (edges_pred, edges_gt)


def get_surface_distance(seg_pred, seg_gt, distance_metric="euclidean"):
    """ (from MONAI)
    This function is used to compute the surface distances from `seg_pred` 
    to `seg_gt`.
    
    Args:
        seg_pred: the edge of the predictions.
        seg_gt: the edge of the ground truth.
        distance_metric: : ["euclidean", "chessboard", "taxicab"]
            the metric used to compute surface distance. Defaults to "euclidean".
            - "euclidean", uses Exact Euclidean distance transform.
            - "chessboard", uses `chessboard` metric in chamfer type transform.
            - "taxicab", uses `taxicab` metric in chamfer type transform.
    Note:
        If seg_pred or seg_gt is all 0, may result in nan/inf distance.
    """
    if isinstance(seg_pred, torch.Tensor):
        seg_pred = seg_pred.detach().cpu().numpy()
    if isinstance(seg_gt, torch.Tensor):
        seg_gt = seg_gt.detach().cpu().numpy()

    if not np.any(seg_gt):
        dis = np.inf * np.ones_like(seg_gt)
    else:
        if not np.any(seg_pred):
            dis = np.inf * np.ones_like(seg_gt)
            return np.asarray(dis[seg_gt])
        if distance_metric == "euclidean":
            dis = distance_transform_edt(~seg_gt)  # invert
        elif distance_metric in {"chessboard", "taxicab"}:
            dis = distance_transform_cdt(~seg_gt, metric=distance_metric)
        else:
            msg = f"distance_metric {distance_metric} is not implemented."
            raise ValueError(msg)

    return np.asarray(dis[seg_pred])


def generate_spatial_bounding_box(
        img, 
        select_fn=None,
        channel_indices=None,
        margin=0
        ):
    """
    generate the spatial bounding box of foreground in the image with 
        start-end positions.
    Users can define arbitrary function to select expected foreground from the 
        whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:
        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]
    The bounding boxes edges are aligned with the input image edges.
    This function returns [-1, -1, ...], [-1, -1, ...] if there's no positive intensity.
    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 
            value provided, use it for all dims.
    """
    def is_positive(img):
        return img > 0

    if select_fn is None:
        select_fn = is_positive
    
    data = img[list(channel_indices)] if channel_indices is not None else img
    data = np.any(select_fn(data), axis=0)
    ndim = len(data.shape)
    margin = margin
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(itertools.combinations(reversed(range(ndim)), ndim - 1)):
        dt = data.any(axis=ax)
        if not np.any(dt):
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        min_d = max(np.argmax(dt) - margin[di], 0)
        max_d = max(data.shape[di] - max(np.argmax(dt[::-1]) - margin[di], 0), min_d + 1)
        box_start[di], box_end[di] = min_d, max_d

    return box_start, box_end