
import os
import re
import pathlib 
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize as nd_resize
import numpy as np
import SimpleITK as sitk

from collections.abc import Sequence



def natural_sort(l):
    """ Given a list of filenames, returns a natural-sorted list. """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def save_image_array(image, file, mask=None):
    """ Image and mask are 3D numpy arrays """
    
    assert image.ndim == 3   
    mask_mult = 1
    if mask is not None:
        assert image.shape == mask.shape
        mask_mult = 2
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(mask_mult, 3, 1)
    ax.imshow(image[image.shape[0] // 2, :, :],
              cmap='gray')
    ax.set_title(f'Axial Slice #{image.shape[0] // 2} (tot: {image.shape[0]})')
    ax = fig.add_subplot(mask_mult, 3, 2)
    ax.imshow(nd_resize(image[:, image.shape[1] // 2, :], (512, 512)),
              cmap='gray')
    ax.set_title(f'Coronal Slice #{image.shape[1] // 2} (tot: {image.shape[1]})')
    ax = fig.add_subplot(mask_mult, 3, 3)
    ax.imshow(nd_resize(image[:, :, image.shape[2] // 2], (512, 512)),
              cmap='gray')
    ax.set_title(f'Sagital Slice #{image.shape[2] // 2} (tot: {image.shape[2]})')
    
    if mask is not None:
        mask = mask.copy() * (255 // mask.max())
        
        ax = fig.add_subplot(mask_mult, 3, 4)
        ax.imshow(mask[mask.shape[0] // 2, :, :],
                cmap='gray')
        ax.set_title(f'Axial Slice #{mask.shape[0] // 2} (tot: {mask.shape[0]})')
        ax = fig.add_subplot(mask_mult, 3, 5)
        ax.imshow(nd_resize(mask[:, image.shape[1] // 2, :], (512, 512)),
                cmap='gray')
        ax.set_title(f'Coronal Slice #{mask.shape[1] // 2} (tot: {mask.shape[1]})')
        ax = fig.add_subplot(mask_mult, 3, 6)
        ax.imshow(nd_resize(mask[:, :, mask.shape[2] // 2], (512, 512)),
                cmap='gray')
        ax.set_title(f'Sagital Slice #{mask.shape[2] // 2} (tot: {mask.shape[2]})')
        
    plt.axis('off') 
    plt.savefig(str(file), bbox_inches='tight')
    plt.close()