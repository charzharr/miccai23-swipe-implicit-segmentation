""" Module utils/io/images2d.py 
Contains common utilities specifor 2D images.

TODO:
 - Cleanup API (combined 2 old utilities files) & format inconsistencies. 
 - Use to_np inside of get_info() (done manually for now)
 - to_file tensor - normalize? unsure if differentiates btwn L and RGB
"""

import sys, os
import shutil
import math
import time
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import cv2
import numpy as np
import PIL.Image as Image, PIL.ImageStat as ImageStat
from skimage.transform import resize as resize
import torch
from torchvision import transforms
import pandas as pd

from ..io import files


### Constants ###
IMAGE_EXTS = ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif',
              '.nii', '.nii.gz', '.dcm']
CONTAINER_TYPES = ['np', 'pil', 'tensor']
COLOR_MODES = ['auto', 'rgb', 'gray', 'binary']
COLOR_2_PILMODE = {
    'rgb': 'RGB', 'gray': 'L', 'binary': '1'
}
PILMODE_2_NUMCHANNELS = {  # Note: GIFs (P) are handled as RGB by default
    '1': 1, 'L': 1, 'I': 1, 'F': 1,   # includes 1 bit, 8 bit, 32 bit pix
    'RGB': 3, 'YCbCr': 3, 'LAB': 3, 'HSV': 3, 'P': 3, # all 3x8-bit pixels
    'RGBA': 4, 'CMYK': 4  # all 4x8-bit pixels
}


### ======================================================================== ###
### * ### * ### * ### *     Module API Implementation    * ### * ### * ### * ###
### ======================================================================== ###

__all__ = ['create_dirs_from_file',
           'is_image', 
           'list_images', 'list_files', 'list_dirs', 'list_all_files_recursive',
           'get_ext', 'get_filename',
           'natural_sort']

# # # # Primary API Functions # # # #
# Note: Generally, imgobj for these can be paths, np, pil, or tensors
#       Also, visualize and get_info only ones that can take in 4D imgobjs


### ---- ### ---- \\    Image Read/Write     // ---- ### ---- ###

def write_np_gray2d(np_im, path, extra_channel=False):
    """ Save 2D grayscale image of shape 1xHxW, HxW, or HxWx1
    """
    if extra_channel:
        np_im = np_im.squeeze(0).squeeze(-1)
    assert np_im.ndim == 2, f"Only 2D gray images are supported."
    img = sitk.GetImageFromArray(np_im, isVector=False)
    _write_image(img, path)


def write_np_rgb2d(np_im, path, channel_last=True):
    """ Save 2D grayscale image of shape 1xHxW, HxW, or HxWx1
    """
    if not channel_last:
        np_im = to_np_channel_last(np_im)
    assert np_im.ndim == 3 and np_im.shape[-1] == 3, \
        f"Only 2D RGB images are supported."
    img = sitk.GetImageFromArray(np_im, isVector=True)
    _write_image(img, path)


def write_np_rgba2d(np_im, path, channel_last=True):
    if channel_last:
        np_im = to_np_channel_first(np_im)
    C = np_im.shape[0]
    channels = [sitk.GetImageFromArray(np_im[i, :, :]) for i in range(C)]
    filter = sitk.ComposeImageFilter()
    img = filter.Execute(*[channels[i] for i in range(C)])
    _write_image(img, path)


def _write_image(img, path, compress=False):
    r""" Creates directory structure and writes using sitk's writer. """ 
    files.create_dirs_from_file(path)
    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(path)
        writer.SetUseCompression(compress)
        writer.Execute(img)
        return True
    except:
        print(f"Error writing image to {path}!")
        return False
    
    
### ---- ### ---- \\    Image Transforms     // ---- ### ---- ###

def to_np_channel_last(np_im):
    return np.moveaxis(np_im, 0, -1)


def to_np_channel_first(np_im):
    return np.moveaxis(np_im, -1, 0)


def to_np(imgobj, size=None, color=None, extra_channel=False):
    """
    (1) np:     RGB HxWx3, Gray/Bin HxW
    (2) pil:    RGB 'RGB', Gray 'L', Bin '1'
    (3) tensor: RGB 3xHxW, Gray/bin HxW

    is_single_channel = False  # assumes RGB
    if color == 'auto':
        img = Image.open(filepath).convert('RGB')
        is_single_channel = is_grayscale(img)
    elif 'bin' in color or 'gray' in color:
        is_single_channel = True
    """
    img = None
    if isinstance(imgobj, str):
        img = _read_img(imgobj, container='np')
    elif isinstance(imgobj, np.ndarray):  
        img = imgobj
    elif 'PIL.' in str(type(imgobj)):  
        img = np.uint8(np.array(imgobj))
    elif isinstance(imgobj, torch.Tensor):
        if torch.min(imgobj) >= 0 and torch.max(imgobj) <= 1:
            imgobj *= 255.
        elif torch.min(imgobj) < 0 or torch.max(imgobj) > 255.:
            print(f"to_np WARNING: {imgobj.shape} tensor has min ({torch.min(imgobj)}) and max ({torch.max(imgobj)})")
        if len(imgobj.shape) == 3:
            if imgobj.shape[0] == 1 or imgobj.shape[0] == 3:
                imgobj = imgobj.permute(1,2,0)
        img = np.uint8(imgobj.cpu().numpy())
    else:
        raise TypeError(f"Image obj of type ({type(imgobj)}) is not valid.")
    
    if size:
        img = _resize(img, size)
    if color:
        img = _recolor(img, color, extra_channel=extra_channel)
    return img


def to_tensor(imgobj, color=None, size=None,
              extra_channel=False):
    
    img = None
    if isinstance(imgobj, str):
        img = _read_img(imgobj, container='ten')
    elif isinstance(imgobj, np.ndarray):
        img = torch.from_numpy(imgobj).float()/255.
        if img.ndim == 3:
            img = img.permute(2,0,1)
    elif 'PIL.' in str(type(imgobj)):
        if imgobj.mode == 'L':
            img = transforms.ToTensor()(imgobj)[0,:,:]  # adds extra dim when L
        else:
            img = transforms.ToTensor()(imgobj)
    elif isinstance(imgobj, torch.Tensor):
        img = imgobj
    else:
        raise TypeError(f"Image obj of type ({type(imgobj)}) is not valid.")
    
    if size:
        img = _resize(img, size)
    if color:
        img = _recolor(img, color, extra_channel=extra_channel)
    return img


def to_pil(imgobj, color=None, size=None):
    
    img = None
    if isinstance(imgobj, str):
        img = _read_img(imgobj, container='pil')
    elif isinstance(imgobj, np.ndarray):
        if imgobj.ndim == 3 and imgobj.shape[-1] == 1:  # HxWx1 not supported
            img = Image.fromarray(imgobj[...,0]).convert('L')  
        else:
            if imgobj.ndim == 3:
                img = Image.fromarray(imgobj).convert('RGB')
            else:
                img = Image.fromarray(imgobj).convert('L')
    elif 'PIL.' in str(type(imgobj)):
        img = imgobj
    elif isinstance(imgobj, torch.Tensor):
        imgobj = to_np(imgobj)
        if imgobj.ndim == 3 and imgobj.shape[-1] == 1:  # HxWx1 not supported
            img = Image.fromarray(imgobj[...,0]).convert('L')  
        else:
            if imgobj.ndim == 3:
                img = Image.fromarray(imgobj).convert('RGB')
            else:
                img = Image.fromarray(imgobj).convert('L')
    else:
        raise TypeError(f"Image obj of type ({type(imgobj)}) is not valid.")
    
    if size:
        img = _resize(img, size)
    if color:
        img = _recolor(img, color)
    return img


def to_file(imgobj, path, color=None, size=None, override=False, mkdir=True):
    r"""
    Saves imgobj to path (image file).

    Parameters:
        imgobj - np array, pil obj, tensor representing a single image
        path - full path including the image name and extension
    """
    # Check assumptions: save dir is valid, no file exists already, etc.
    if os.path.isfile(path):
        if override:
            print(f"(to_file) overriding file '{path}'")
        else:
            raise IOError(f"File ({path}) already exists & override is off")
    if not os.path.isdir(os.path.dirname(path)):
        if mkdir:
            print(f"(to_file) making directory for image's path "
                  f"'{os.path.dirname(path)}'")
            Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        else:
            raise IOError(f"Directory ({os.path.dirname(path)}) does not "
                          f"exist and mkdir is set to False.")
    
    img = None  # get input img as pil object
    if isinstance(imgobj, str):
        assert files.is_image(imgobj), f'Invalid image file given ({imgobj})'
        if files.get_ext(imgobj) == files.get_ext(path):
            shutil.copy(imgobj, path)  # overrides by default, checks above
            return
        else:  # image format is different, needs conversion
            img = Image.open(imgobj)
    elif isinstance(imgobj, np.ndarray):
        dims = imgobj.shape
        assert len(imgobj.shape) == 2 or len(imgobj.shape) == 3
        if len(dims) == 3:
            assert dims[-1] == 1 or dims[-1] == 3
            if dims[-1] == 1:
                img = Image.fromarray(imgobj).convert('L')
            else:
                img = Image.fromarray(imgobj).convert('RGB')
        else:  # 2D grayscale HxW image
            img = Image.fromarray(imgobj).convert('L') # assumed range: 0 to 255
    elif 'PIL.' in str(type(imgobj)):
        img = imgobj
    elif isinstance(imgobj, torch.Tensor):
        img = transforms.ToPILImage()(imgobj)
    else:
        raise TypeError(f"Image obj of type ({type(imgobj)}) is not valid.")

    if size:
        img = _resize(img, size)
    if color:
        img = _recolor(img, color)
    print(f"(to_file) writing image to '{path}'")
    img.save(path)
    return path


### ---- ### ---- \\   Image Queries, Listing, Properties   // ---- ### ---- ###


def get_info(imgobjs, verbose=True):
    r"""
    Print basic info of image or image stack of filepath, np, pil or tensor.
        > Assumes images in stack or list have same number of channels.

    Parameters:
        imgobjs - individual img (str,np,pil,ten) or stack (strs,4dnp,pils,4dten)
        verbose - prints statistics for each individual image instance
        std     - display standard deviation over all images for each channel
    Returns: the dimensions as B, C, H, W.
    """

    ## Put entries in list format & figure out N
    N, formatted_imgobjs = 0, []
    display_exts = False
    if isinstance(imgobjs, list):
        for io in imgobjs:
            formatted_imgobjs += _create_img_list(io)
            if isinstance(io, str):
                display_exts = True
    else:
        formatted_imgobjs = _create_img_list(imgobjs)
    N = len(formatted_imgobjs)
    assert N > 0

    print(f"\nCollecting information from {N} images..\n", flush=True)
    if verbose and N > 1:
        print(f"- - - - - - -\nInstance Info\n- - - - - - -")

    ## Collect Info
    # Globals
    gray_counter = 0
    dims_2_count, exts_2_count = Counter(), Counter()
    sum_sizes = [0., 0.]  # H, W
    # RGB
    rgb_absmin, rgb_absmax = [0., 0., 0.], [0., 0., 0.]
    sum_rgb_minvals, sum_rgb_maxvals = [0., 0., 0.], [0., 0., 0.]
    sum_rgb_means, sum_rgb_stds = [0., 0., 0.], [0., 0., 0.]
    # gray
    gray_absmin, gray_absmax = 0., 0.
    sum_gray_minvals, sum_gray_maxvals = 0., 0.
    sum_gray_means, sum_gray_stds = 0., 0.
    
    rgb_initialized, gray_initialized = False, False

    iterable = formatted_imgobjs if verbose else tqdm(formatted_imgobjs)
    for i, img in enumerate(iterable):
        minval, maxval = [], []
        mean, std = [], []
        dim, ext = None, None
        
        np_img = to_np(img)
        
        is_gray = True
        if np_img.ndim == 3:
            is_gray = False if np_img.shape[-1] == 3 else True
            for c in range(np_img.shape[-1]):
                minval.append(np.min(np_img[:,:,c]))
                maxval.append(np.max(np_img[:,:,c]))
                mean.append(np.mean(np_img[:,:,c]))
                std.append(np.std(np_img[:,:,c]))
        else:
            minval.append(np.min(np_img[:,:]))
            maxval.append(np.max(np_img[:,:]))
            mean.append(np.mean(np_img[:,:]))
            std.append(np.std(np_img[:,:]))

                
        
        # Update globals for current image
        if is_gray: 
            gray_counter += 1
            if not gray_initialized:
                gray_initialized = True
                gray_absmin = minval[0]
                gray_absmax = maxval[0]
            else:
                if minval[0] < gray_absmin:
                    gray_absmin = minval[0]
                if maxval[0] > gray_absmax:
                    gray_absmax = maxval[0]
            sum_gray_minvals += minval[0]
            sum_gray_maxvals += maxval[0]
            sum_gray_means += mean[0]
            sum_gray_stds += std[0]
        else:  # is rgb
            if not rgb_initialized:
                rgb_initialized = True
                rgb_absmin = minval
                rgb_absmax = maxval
            else:
                for c in range(len(minval)):
                    if minval[c] < rgb_absmin[c]:
                        rgb_absmin[c] = minval[c]
                    if maxval[c] > rgb_absmax[c]:
                        rgb_absmax[c] = maxval[c]
            for c in range(len(minval)):
                sum_rgb_minvals[c] += minval[c]
                sum_rgb_maxvals[c] += maxval[c]
                sum_rgb_means[c] += mean[c]
                sum_rgb_stds[c] += std[c]
        dim = get_dimensions(np_img)
        dims_2_count[dim] += 1
        sum_sizes[0] += dim[2]
        sum_sizes[1] += dim[3]
        if isinstance(img, str):
            ext = files.get_ext(img)
            exts_2_count[ext] += 1
        
        # print individual instance info
        if verbose and N > 1:
            print(f"> Image #{i+1}/{N}. Size: {dim}.")
            if isinstance(img, str):
                print(f"\tFile: {files.get_filename(img)}")
            print(f"\tMeans:  {mean} \n\tStds:   {std}\n"
                  f"\tMins:   {minval} \n\tMaxes:  {maxval}")
    
    ## Print Global Info
    N_RGB = N - gray_counter

    print(f"\n- - - - - -\nGlobal Info\n- - - - - -")
    print(f"Num Images:  {N} ({N-gray_counter} RGB, {gray_counter} Grayscale)")
    print(f"Average Size: {int(sum_sizes[0]/N)}x{int(sum_sizes[1]/N)}")
    if display_exts:
        print(f"\tFile Types & Counts..")
        for counts in exts_2_count.most_common(len(exts_2_count)):
            print(f"\t\t{counts[0]}\t{counts[1]}")
    print(f"\tImage Sizes & Counts..")
    for counts in dims_2_count.most_common(len(dims_2_count)):
        print(f"\t\t{counts[0]}\t{counts[1]}")

    if N_RGB > 0:
        print(f"\n[RGB Images]")
        print(f"Avg Maxes:  {[t/N_RGB for t in sum_rgb_maxvals]}\n"
              f"Abs Maxes:  {rgb_absmax}")
        print(f"Avg Minis:  {[t/N_RGB for t in sum_rgb_minvals]}\n"
              f"Abs Minis:  {rgb_absmin}")
        print()
        rgb_means = [t/N_RGB for t in sum_rgb_means]
        print(f"Norm-Means: {rgb_means} \n"
              f"Abs Mean:    {sum(rgb_means)/3}")
        rgb_stds = [t/N_RGB for t in sum_rgb_stds]
        print(f"Norm-Stds:  {rgb_stds}\n"
              f"Abs Std:     {sum(rgb_stds)/3}") 
    
    if gray_counter > 0:
        print(f"\n[Gray Images]")
        print(f"Avg Max:     {sum_gray_maxvals/gray_counter}\n"
              f"Abs Max:     {gray_absmax}")
        print(f"Avg Mini:    {sum_gray_minvals/gray_counter}\n"
              f"Abs Mini:    {gray_absmin}")
        print()
        print(f"Norm-Mean:   {sum_gray_means/gray_counter} ")
        print(f"Norm-Std:    {sum_gray_stds/gray_counter} ")


def collate(imgobjs, container='np', color=None, size=None):
    r"""
    Takes in list (img objs) and collates the data into 4D stack (np/tensor).

    Parameters:
        imgobjs - list of filenames, path(s), pil objects, 
                  3d/4d tensors/np
        container - can be np stack or tensor stack from np
        color - rgb or single channel gray (Bx1xHxW)
        size - resized image (if not given, defaults to avg image size)
    """
    
    ## collects objects into list and dimensions
    imgs = []
    if isinstance(imgobjs, list):
        for io in imgobjs:
            imgs += _create_img_list(io)
    else:
        imgs = _create_img_list(imgobjs)

    ## zero-initiated placeholder stack 
    channels = 0
    if not size:
        size = [0, 0]  # H, W
        for img in imgs:
            _, C, H, W = get_dimensions(img)
            size[0] += H
            size[1] += W
            channels = C if C > channels else channels
        size = [int(d/len(imgs)) for d in size]
    assert len(size) == 2 and size[0] > 0 and size[1] > 0

    if color:
        channels = 3 if color == 'rgb' else 1
    else:
        color = 'rgb' if channels == 3 else 'gray'
    
    ## create and return stack
    if container == 'np':
        stack = np.zeros((len(imgs),size[0],size[1],channels))
        for i, img in enumerate(imgs):
            img = to_np(img, size=size, color=color, extra_channel=True)
            stack[i,...] = img
        return stack
    else:
        stack = torch.zeros((len(imgs),channels,size[0],size[1]))
        for i, img in enumerate(imgs):
            stack[i,:,:,:] = to_tensor(img, size=size, color=color)
        return stack
    

def get_dimensions(imgobjs):
    r"""  *4D stack accepted for matrix representations only*

    Returns B,C,H,W for a single image (file, np, pil, tens) or 4D (np, tens).
        Note: for file paths, this process involves MSE calculating.

    Parameters
        imgobjs - can be a filepath, pil object, np image (BxHxWxC),
                  or torch tensor (BxCxHxW)
    """

    num_images, num_channels, num_rows, num_cols = 1, 3, 0, 0
    if isinstance(imgobjs, str) or 'PIL.' in str(type(imgobjs)):
        # PIL.open ~10x faster than imread to np
        if isinstance(imgobjs, str):
            img = _read_img(imgobjs, container='pil')
        else:
            img = imgobjs
        num_cols, num_rows = img.size
        if len(img.getbands()) == 1:
            num_channels = 0
    elif isinstance(imgobjs, np.ndarray):
        assert imgobjs.ndim >= 2 and imgobjs.ndim <= 4, \
               f"Given numpy shape ({imgobjs.shape}) doesn't describe an image."
        shape = imgobjs.shape
        if len(shape) == 4:  # 4D stack
            num_images, num_rows, num_cols, num_channels = imgobjs.shape
        elif len(shape) == 3:  # assumed 1 RGB image
            assert shape[-1] == 3 or shape[-1] == 1
            num_rows, num_cols, num_channels = shape
        else:  # 2 ndims
            num_rows, num_cols = shape
            num_channels = 0
    elif isinstance(imgobjs, torch.Tensor):
        assert imgobjs.ndim >= 2 and imgobjs.ndim <= 4, \
               f"Given tensor shape ({imgobjs.shape}) doesn't describe an image."
        size = imgobjs.size()
        if len(size) == 4:  # 4D stack
            num_images, num_channels, num_rows, num_cols = size
        elif len(size) == 3:  # assumed 1 RGB image
            assert size[0] == 1 or size[0] == 3
            num_channels, num_rows, num_cols = size
        else:
            num_rows, num_cols = size
            num_channels = 0
    else:
        raise TypeError(f"Image obj of type ({type(imgobjs)}) is not valid.")

    return num_images, num_channels, num_rows, num_cols


def get_mean_std_norms(imgobjs, use_float=False):
    r"""
    Returns the average means & stds for each image channel.

    Parameters
        imgobjs - can be list (or single) of 3d/4d np/tens, pils, files/paths
        use_float - true transforms image intensities to range [0,1]
    """
    
    ## Put entries in list format & figure out N
    N, formatted_imgobjs = 0, []
    if isinstance(imgobjs, list):
        for io in imgobjs:
            formatted_imgobjs += _create_img_list(io)
    else:
        formatted_imgobjs = _create_img_list(imgobjs)
    N = len(formatted_imgobjs)
    assert N > 0

    ## calculates running sums of means/stds
    sum_means, sum_stds = [0., 0., 0.], [0., 0., 0.]
    for i, img in enumerate(formatted_imgobjs):
        mean, std = [0., 0., 0.], [0., 0., 0.]  # if gray, updates first val
        np_img = to_np(img)
        if use_float:
            np_img /= 255.
        
        if np_img.ndim == 3:
            for c in range(np_img.shape[-1]):
                mean[c] = np.mean(np_img[:,:,c])
                std[c] = np.std(np_img[:,:,c])
        else:
            mean[0] = np.mean(np_img[:,:])
            std[0] = np.std(np_img[:,:])

        for c in range(3):
            sum_means[c] += mean[c]
            sum_stds[c] += std[c]

    return [m/N for m in sum_means], [s/N for s in sum_stds]


# # # # Secondary Functions # # # #
# Note: these don't worry about image paths as imgobj

def is_grayscale(imgobj, thumbsize=40, MSEthreshold=22, adjust_color_bias=False):
    r"""  *Note: this is a slow, hard check that is data dependent!
    Checks if given image (filepath, pil, np, tensor) is grayscale.
    (1) Converts to RGB  (2) Computer average channel-variance per pixel
    
    Parameters
        imgobj - can be filepath, np, pil, tens
        thumbsize - downsized image to be analyzed (default 40x40)
        MSEthreshold - considered binary if MSE channel vals <= this
        adjust_color_bias - subtracts every value from mean value (unnecessary)
    """
    debug = False

    pil_img = imgobj if 'PIL.' in str(type(imgobj)) else to_pil(imgobj)
    if pil_img.getbands()[0] == 'P':
        pil_img = pil_img.convert('RGB')
    
    bands = pil_img.getbands()
    if bands == ('R','G','B') or bands == ('R','G','B','A'):
        thumb = pil_img.resize((thumbsize,thumbsize))
        SSE, bias = 0, [0,0,0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias)/3 for b in bias]
        for pixel in thumb.getdata(): # pixel = (R,G,B)
            mu = sum(pixel)/3
            SSE += sum((pixel[i] - mu - bias[i])**2 for i in [0,1,2])
        MSE = SSE/(thumbsize * thumbsize)
        if debug:
            print(f"MSE = {MSE:.4f}")
        if MSE <= MSEthreshold:
            return True
        else:
            return False
    elif len(bands) == 1:
        if debug:
            print(f"Only 1 band (black and white pic).")
        return True
    else:
        raise TypeError(f"PIL bands ({bands}) is not supported.")


def normalize(imgobj, means=None, stds=None, container='np'):
    
    if not means and not stds:
        means, stds = get_mean_std_norms(imgobj)
    elif not means and stds:
        means, _ = get_mean_std_norms(imgobj)
    elif means and not stds:
        _, stds = get_mean_std_norms(imgobj)
    assert len(means) == 3 and len(stds) == 3

    if container == 'np':
        img = to_np(imgobj)


def unnormalize(imgobj, means=None, stds=None, container='np'):
    pass





### ======================================================================== ###
### * ### * ### * ### *        Module API Helpers        * ### * ### * ### * ###
### ======================================================================== ###


def _read_img(filepath, container='np'):
    r"""
    Helper function that coverts image path to [R G B] data (even if gray/bin).
    Does not handle color or size conversions.
        (This is called by to_np/pil/tensor when a path to 1 file is given.)

    Parameters
        filepath - full (or relative) path to image file
        container - output type (can be: np, pil, tensor)
    """
    assert files.is_image(filepath, checkfile=True), \
           f"File ({filepath}) is not a valid img."
    
    pil_img = Image.open(filepath)
    if pil_img.getbands()[0] == 'P':
        # print('P detected: ', filepath)
        if is_grayscale(pil_img):
            pil_img = pil_img.convert('L')
        else:
            pil_img = pil_img.convert('RGB')
        # print(pil_img.mode, pil_img.getbands(), np.array(pil_img).shape)
    img_bands = pil_img.getbands()

    img = None  # to be set by container conditions
    container = container.lower()
    if container == 'np':
        img = np.uint8(np.array(pil_img))
        # img = cv2.imread(filepath)[...,::-1]  # (H,W,bgr) -> (H,W,rgb)
        # assert img is not None, f"Error reading image ({filepath})"
        # if str(img.dtype) != 'uint8':
        #     img = np.uint8(img)
    elif container == 'pil':
        img = pil_img
    elif 'ten' in container:
        # if files.get_ext(filepath) == '.gif':  # ~3-4x slower than cv2
        #     npimg = np.array(Image.open(filepath).convert('RGB'))  
        # else:
        #     # still ~8-10x faster to copy neg strides than transform
        #     npimg = cv2.imread(filepath)[...,::-1].copy() # (H,W,bgr) -> (H,W,rgb)
        npimg = np.array(pil_img)
        if len(img_bands) > 1:  # >10x faster than PIL transform
            img = torch.from_numpy(np.rollaxis(npimg, 2, 0))
        else:
            img = torch.from_numpy(npimg)
        img = img.float()/255.
    else:
        raise TypeError(f"Container type given ({container}) is not valid.")

    return img


def _resize(imgobj, size=None):
    r"""
    Helper function that resizes single np, pil, or tens image

    Parameters:
        size - (H,W)
    """
    if not size:
        return imgobj
    assert len(size) == 2
    
    if isinstance(imgobj, np.ndarray):  # standardizes array
        return resize(imgobj, (size[0], size[1]))
    elif 'PIL.' in str(type(imgobj)):  
        return imgobj.resize(size[::-1])
    elif isinstance(imgobj, torch.Tensor):
        # if size[0] == size[1]:
        #     return F.interpolate(imgobj, size=size[0])
        # else:
        resized_pil = transforms.ToPILImage()(imgobj).resize(size[::-1])
        return transforms.ToTensor()(resized_pil)
    else:
        raise TypeError(f"Image obj of type ({type(imgobj)}) is not valid.")
    

def _recolor(imgobj, color=None, extra_channel=False):
    r"""
    Helper function that recolors single np, pil, or tens image

    Parameters:
        color - auto, rgb, gray
        extra_channel - for gray  e.g. np: (H,W) without and (H,W,1) with
    """
    if not color or color == 'auto':
        return imgobj
    
    if isinstance(imgobj, np.ndarray):
        # print('_recolor: ', imgobj.shape)
        assert imgobj.ndim == 3 or imgobj.ndim == 2
        if color == 'rgb':
            if imgobj.ndim == 3 and imgobj.shape[-1] == 3:
                return imgobj
            elif imgobj.ndim == 3 and imgobj.shape[-1] == 1:
                return np.repeat(imgobj, 3, axis=2)
            else:
                return np.repeat(imgobj[:,:,np.newaxis], 3, axis=2)
        else:  # convert to gray
            if imgobj.ndim == 3 and imgobj.shape[-1] == 1:
                if extra_channel:
                    return imgobj
                return imgobj[:,:,0]
            elif imgobj.ndim == 2:
                if extra_channel:
                    return np.repeat(imgobj[:,:,np.newaxis], 1, axis=2)
                return imgobj
            else:  # 3 channel rgb
                # return np.dot(imgobj[...,:3], [0.299, 0.587, 0.114])
                imgobj = np.dot(imgobj, [0.299, 0.587, 0.114])
                if extra_channel:
                    return np.repeat(imgobj[:,:,np.newaxis], 1, axis=2)
                return imgobj
    elif 'PIL.' in str(type(imgobj)):  
        return imgobj.convert(COLOR_2_PILMODE[color])
    elif isinstance(imgobj, torch.Tensor):
        assert imgobj.ndim == 3 or imgobj.ndim == 2
        if color == 'rgb':
            if imgobj.ndim == 3 and imgobj.shape[0] == 3:  # 3xHxW
                return imgobj
            else:  # 1xHxW or HxW
                return imgobj.repeat(3,1,1)
        else:  # convert to gray
            if imgobj.ndim == 3 and imgobj.shape[0] == 1:
                if extra_channel:
                    return imgobj
                return imgobj[0,:,:]
            elif imgobj.ndim == 2:
                if extra_channel:
                    return imgobj.repeat(1,1,1)
                return imgobj
            else:  # 3 channel rgb
                # return np.dot(imgobj[...,:3], [0.299, 0.587, 0.114])
                imgobj = .299*imgobj[0,:,:] + .587*imgobj[1,:,:] + .114*imgobj[2,:,:]
                if extra_channel:
                    return imgobj.repeat(1,1,1)
                return imgobj
    else:
        raise TypeError(f"Image obj of type ({type(imgobj)}) is not valid.")


def _create_img_list(imgobj):
    r"""
    Returns list of single images in file/path, 3d/4d np/tensor, pil
    """

    imgs = []
    if isinstance(imgobj, str):
        if os.path.isdir(imgobj):
            imgs = files.list_images(imgobj, fullpath=True)
        elif os.path.isfile(imgobj) and files.is_image(imgobj):
            imgs = [imgobj]
        else:
            print(f"WARNING: Path ({imgobj}) is not valid.")
    elif 'PIL.' in str(type(imgobj)):
        imgs = [imgobj]
    elif isinstance(imgobj, np.ndarray) or isinstance(imgobj, torch.Tensor):
        assert imgobj.ndim >= 2 and imgobj.ndim <= 4
        if imgobj.ndim == 4:
            for c in range(imgobj.shape[0]):
                imgs.append(imgobj[c,:,:,:])
        else:
            imgs = [imgobj]
    elif isinstance(imgobj, pd.DataFrame):
        assert 'path' in imgobj.columns
        for path in imgobj['path']:
            imgs.append(path)
    else:
        raise TypeError(f"Image obj of type ({type(imgobj)}) is not valid.")
    
    return imgs
