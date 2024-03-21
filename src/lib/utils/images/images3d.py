""" Module utils/io/images3d.py 
Contains common utilities for 3D images.
"""

import sys, os
import SimpleITK as sitk
import numpy as np
import torch

from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates

from ..io import files as file_utils

### Constants ###
IMAGE_EXTS = ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif',
              '.nii', '.nii.gz', '.dcm']


### ======================================================================== ###
### * ### * ### * ### *          API Definitions         * ### * ### * ### * ### 
### ======================================================================== ###

__all__ = ['read_sitk3d',
           'write_sitk_gray3d', 'write_np_gray3d',
           'to_np_channel_last', 'to_np_channel_first'
          ]

### ---- ### ---- \\    Image Read/Write     // ---- ### ---- ###

def read_sitk3d(im_path, pixtype=sitk.sitkInt16):
    image = sitk.ReadImage(im_path, pixtype)
    dims = image.GetNumberOfComponentsPerPixel()
    return image


def write_sitk_gray3d(sitk_im, path, compress=False):
    r""" Save 3D sitk grayscale image. 
    Args:
        sitk_im: SimpleITK.Image
        path: str or pathlib.Path
        compress: bool
            If save-type is .nii or .nii.gz, doesn't affect output.
    """
    channels = sitk_im.GetNumberOfComponentsPerPixel()
    assert channels == 1, f"Only 3D gray sitk images are supported."
    _write_image(sitk_im, path, compress=compress)


def write_np_gray3d(np_im, path, extra_channel=False, compress=False):
    r""" Save 3D grayscale image of shape 1xHxWxD, HxWxD, or HxWxDx1 """
    if extra_channel:
        np_im = np_im.squeeze(0).squeeze(-1)
    assert np_im.ndim == 3, f"Only 3D gray images are supported."
    img = sitk.GetImageFromArray(np_im, isVector=False, compress=False)
    _write_image(img, path, compress=compress)


def _write_image(img, path, compress=False):
    r""" Creates directory structure and writes using sitk's writer. """ 
    file_utils.create_dirs_from_file(path)
    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(path)
        writer.SetUseCompression(compress)
        writer.Execute(img)
        return True
    except:
        print(f"Error writing image to {path}!")
        file_utils.delete_file(path)
        return False
        

### ---- ### ---- \\    Image Structural Transforms     // ---- ### ---- ###


def sitk_to_np(sitk_im):
    return sitk.GetArrayFromImage(sitk_im)


def np_to_sitk(np_im, has_channels=False):
    r"""
    Args:
        has_channels: bool
            If image is rgb, has_channels=True
    """
    return sitk.GetImageFromArray(np_im, isVector=has_channels)
    

def to_np_channel_last(np_im):
    if np_im.shape[0] not in range(1, 6):
        print(f"(to_np_channel_last) WARNING: Weird image {np_im.shape}")
    return np.moveaxis(np_im, 0, -1)


def to_np_channel_first(np_im):
    if np_im.shape[-1] not in range(1, 6):
        print(f"(to_np_channel_first) WARNING: Weird image {np_im.shape}")
    return np.moveaxis(np_im, -1, 0)


### ---- ### ---- \\    Volumetric Image Preprocessing     // ---- ### ---- ###

def is_anisotropic(sitk_image):
    """ Returns if 3D img is anisotropic according to the NNUNet definition."""
    spacing = np.array(sitk_image.GetSpacing()).tolist()
    min_space = min(spacing)
    max_space = max(spacing) 
    is_anisotropic = True if max_space > 3 * min_space else False 
    return is_anisotropic
    

def resize_segmentation3d(np_tens_mask, new_shape, order=3, class_ids=[]):
    """ ~40 seconds for a BCV volume. No changes to original mask data.
    1. Uses one-hot version of mask to resizing with linear interpolation
        per channel to avoid artifacts.
    2. Threshold each channel @ 0.5 to a clean one-hot
    3. Convert back to the format the input was in.

    Scikit Orders from 0 to 5:
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic

    Args:
        mask (array or tensor): one-hot or ID mask
        new_shape (tuple DxHxW): new shape of mask
    """
    assert len(new_shape) == 3, f'"new_shape" must be DxHxW list. {new_shape}'

    is_tensor = isinstance(np_tens_mask, torch.Tensor)
    if is_tensor:
        mask_arr = np_tens_mask.detach().cpu().numpy()
    else:
        mask_arr = np_tens_mask 
    dtype = mask_arr.dtype

    if mask_arr.ndim == 3:  # convert to 1 hot
        if order == 0:
            resized_channel = resize(channel_mask.astype(np.float32),
                                     new_shape, order, mode='edge',
                                     clip=True, anti_aliasing=False)
            final_mask = resized_channel.astype(dtype)
        else:
            unique_labels = np.unique(mask_arr) if not class_ids else class_ids
            final_mask = np.zeros(new_shape, dtype=dtype)
            for i, val in enumerate(unique_labels):
                channel_mask = mask_arr == val
                if channel_mask.sum() > 0:
                    resized_channel = resize(channel_mask.astype(np.float32),
                                            new_shape, order, mode='edge',
                                            clip=True, anti_aliasing=False)
                    final_mask[resized_channel >= 0.5] = val
    else:  # resize each channel of one-hot mask
        assert mask_arr.ndim == 4
        raise NotImplementedError('Not yet tested.')
        final_mask = np.zeros([mask_arr.shape[0]] + list(new_shape), 
                              dtype=mask_arr.dtype)
        for i in range(mask_arr.shape[0]):
            channel_mask = mask_arr[i]
            resized_channel = resize(channel_mask.astype(np.float32),
                                     new_shape, order, mode='edge',
                                     clip=True, anti_aliasing=False)
            final_mask[resized_channel >= 0.5] = 1

    if is_tensor:
        return torch.from_numpy(final_mask)

    return final_mask


def resample_image(orig_sitk_image, new_spacing, anisotropic=False, 
                   verbose=False):
    """ July 2022. Most up to date volumetric image resampling.
    Using NNUNet preprocessing based on spacing.
    """
    orig_spacing = np.array(orig_sitk_image.GetSpacing())
    orig_size = np.array(orig_sitk_image.GetSize())
    
    if orig_spacing.tolist() == new_spacing.tolist():
        if verbose:
            print(f'Spacing & Size Unchanged: {orig_spacing} | {orig_size}')
        return orig_sitk_image
    
    new_size = orig_size * (orig_spacing / new_spacing) 
    new_size = np.ceil(new_size).astype(np.int32).tolist()
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(orig_sitk_image.GetDirection())
    resample.SetOutputOrigin(orig_sitk_image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetInterpolator = sitk.sitkBSpline
    resample.SetSize(new_size)
    
    if anisotropic:
        # 1st resample planar direction 
        resample.SetSize(orig_size[0], new_size)
        

def resample_data_nnunet(data, new_shape, order=3, is_seg=False, class_ids=[],
                         do_separate_z=False, axis=None, order_z=0):
    """
    Args:
        data (np.ndarray): Single image data of shape (C,X,Y,Z)
        new_shape (list): New shape (X,Y,Z)
        order (int): 0=NN, 1=Bi-Linear, 2=Bi-Quadratic, 3=Bi-Cubic)
        is_mask (bool): calls resize_segmentation3d if is_mask
        do_separate_z (bool): set this as true if image is anisotropic
        axis (list): first element is the anisotropic axis
        order_z (int 0-5): order of z axis spline interpolation
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation_nnunet
        kwargs = {'class_ids': class_ids}
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
                orig_shape_2d = shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
                orig_shape_2d = shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]
                orig_shape_2d = shape[:-1]
                
            print(f'Separating z along axis {axis} from shape {new_shape},'
                  f'Data shape: {data.shape}')

            reshaped_final_data = []
            for c in range(data.shape[0]):
                
                # Sample planar
                if np.all(new_shape_2d == orig_shape_2d):
                    reshaped_data = data[c]
                else:
                    reshaped_data = []
                    for slice_id in range(shape[axis]):
                        if axis == 0:
                            reshaped_data.append(resize_fn(data[c, slice_id], 
                                                 new_shape_2d, order, 
                                                 **kwargs).astype(dtype_data))
                        elif axis == 1:
                            reshaped_data.append(resize_fn(data[c, :, slice_id], 
                                                 new_shape_2d, order, 
                                                 **kwargs).astype(dtype_data))
                        else:
                            reshaped_data.append(resize_fn(data[c, :, :, slice_id], 
                                                 new_shape_2d, order, 
                                                 **kwargs).astype(dtype_data))
                    reshaped_data = np.stack(reshaped_data, axis)
                
                # Adjust z
                if shape[axis] != new_shape[axis]:
                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(
                            map_coordinates(reshaped_data, coord_map, 
                                            order=order_z, 
                                            mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), 
                                                coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            # print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, 
                                **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data 
    

def resize_segmentation_nnunet(segmentation, new_shape, order=3, class_ids=[]):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    msg = "New shape must have same dimensionality as segmentation."
    assert len(segmentation.shape) == len(new_shape), msg
    
    unique_labels = class_ids
    if not class_ids:
        unique_labels = np.unique(segmentation)
    
    tpe = segmentation.dtype
    
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", 
                      clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, 
                                       mode="edge", clip=True, 
                                       anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped
