
import colorsys
import scipy
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from ..io import images2d as i2d_utils, images3d as i3d_utils
from . import np_image as npi_utils



### ---- ### ---- \\    3D & 2D Plotting Functions     // ---- ### ---- ### 


def plot_3d_slices(np_volume, mask=None,
                   title='', slices=15, image_size=5, save=''):
    """ Plots sagittal, coronal, and axial slices at centered intervals of img.
    """
    assert np_volume.ndim == 3, f"Only 3D grayscale images HxWxD accepted."
    shape = np_volume.shape
    if mask is not None:
        assert shape == mask.shape, f"Mask and image shapes should match."
        mask_rgb = mask_to_color(mask)
    
    matrix_of_slices, matrix_of_titles = [], []
    for dim in range(np_volume.ndim):
        dim_slices, dim_titles = [], []
        dim_range = range(0, np_volume.shape[dim], np_volume.shape[dim] // slices)
        for dim_index in dim_range:
            if dim == 0:  # axial
                if mask is not None:
                    mslice = mask_rgb[dim_index, ...]
                slice = np_volume[dim_index, :, :]
                title = 'axial: ' + _get_plot_title(slice)
            elif dim == 1:
                if mask is not None:
                    mslice = mask_rgb[:, dim_index, ...]
                slice = np_volume[:, dim_index, :]
                title = 'sagittal: ' + _get_plot_title(slice)
            else:
                if mask is not None:
                    mslice = mask_rgb[:, :, dim_index, ...]
                slice = np_volume[:, :, dim_index]
                title = 'axial: ' + _get_plot_title(slice)
            slice = i3d_utils.hu_to_np_image(slice, clip=[-1024, 325], 
                                             scale=[0, 255], rgb=True)
            if mask is not None:
                slice = np.round(0.3 * mslice + 0.7 * slice).astype(np.uint8)
            dim_slices.append(slice)
            dim_titles.append(title)
        matrix_of_slices.append(dim_slices)
        matrix_of_titles.append(dim_titles)
    npi_utils.print_image_summary(np_volume)
    plot_grid(matrix_of_slices, titles=matrix_of_titles, image_size=image_size,
              save=save)


def plot_2d_image(np_image, title='', image_size=10, save=''):
    assert npi_utils.is_standard_2dimage(np_image), 'Image not standard!'
    axis_title = _get_plot_title(np_image)
    np_image = npi_utils.standardize_2dimage(np_image)
    titles = [[axis_title]] if not title else [[axis_title + ' | ' + title]]
    plot_grid([[np_image]], titles=titles, image_size=image_size, save=save)
    

def plot_2d_images(np_images_list, titles=[], cols=3, image_size=10, save=''):
    if len(np_images_list) == 0:
        return
    if titles:
        assert len(np_images_list) == len(titles)
    
    R = (len(np_images_list) // cols) + 1
    matrix_of_images, matrix_of_titles = [], []
    for r in range(R):
        images_row, titles_row = [], []
        for c in range(cols):
            curr_idx = r * cols + c
            if curr_idx >= len(np_images_list):
                images_row.append(None)
                titles_row.append('')
                continue
            np_image = np_images_list[curr_idx]
            axis_title = _get_plot_title(np_image)
            custom_title = '' if not titles else ' | ' + titles[curr_idx]
            
            assert npi_utils.is_standard_2dimage(np_image), 'Image not standard!'
            
            np_image = npi_utils.standardize_2dimage(np_image)
            axis_title = axis_title + custom_title
            images_row.append(np_image)
            titles_row.append(axis_title)
        matrix_of_images.append(images_row)
        matrix_of_titles.append(titles_row)
    plot_grid(matrix_of_images, titles=matrix_of_titles, 
              image_size=image_size, save=save)


def plot_grid(matrix_of_images, titles=None, image_size=10, save=''):
    R, C = len(matrix_of_images), len(matrix_of_images[0])
    fig = plt.figure(figsize=(C * image_size, R * image_size))
    for r in range(R):
        for c in range(C):
            position = r * C + (c + 1)
            # print(position)
            # print(titles[r][c])
            title = titles[r][c] if titles is not None else ''
            image = matrix_of_images[r][c]
            if image is not None:
                ax = fig.add_subplot(R, C, position)
                ax.set_title(title)
                ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    # plt.subplots_adjust(left=1, bottom=0, right=2, top=1, wspace=0, hspace=0)
    plt.show()
    
    
def _get_plot_title(np_image):
    shape = np_image.shape
    shape_str = 'x'.join([str(s) for s in shape])
    max_val = np_image.max()
    max_str = f"{max_val}" if 'int' in str(type(max_val)) else f"{max_val:.2f}"
    min_val = np_image.min()
    min_str = f"{min_val}" if 'int' in str(type(min_val)) else f"{min_val:.2f}"
    return f"{shape_str} max:{max_str} min:{min_str}"


### ---- ### ---- \\    Segmentation Mask Functions     // ---- ### ---- ###


# def mask_to_color(np_mask, channel_last=True):
#     """
#     Arguments:
#         np_mask - CxHxW(xD) 
#     """
#     if 'bool' not in np_mask.dtype:
#         np_mask = np_mask > 0
#     class_channel = -1 if channel_last else 0
#     num_classes = np_mask.shape[class_channel]
#     rgb_colors = generate_colors(num_classes)
#     new_mask = np.zeros(tuple(list(np_mask.shape) + [3]))
#     for cnum in range(1, num_classes):  # cnum 0 = background
#         new_mask[np_mask == cnum] = rgb_colors[cnum]
#     return new_mask


def mask_to_color(sitk_np_im, has_channels=False):
    if has_channels:  # each class label defined in its own channel.
        raise NotImplementedError()
    np_im = sitk_np_im
    if isinstance(sitk_np_im, sitk.Image):
        np_im = sitk.GetImageFromArray(sitk_np_im)
    assert isinstance(np_im, np.ndarray)
    class_values = np.unique(np_im)
    colors = generate_colors(len(class_values))
    
    colored_mask = np.zeros(list(np_im.shape) + [3]).astype(np.uint8)
    for c in range(len(class_values)):
        colored_mask[np_im == class_values[c]] = colors[c]
    return colored_mask


def overlay_mask(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed


def generate_colors(num_colors, black_first=True):
    if black_first:
        HSVs = [[0, 0, 0]]
        if num_colors > 1:
            HSVs += [[x / (num_colors - 1), 0.5, 0.5] for x in range(num_colors - 1)]
    else:
        HSVs = [[x / num_colors, 0.5, 0.5] for x in range(num_colors)]
    RGBs = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSVs))
    RGBs = [[int(255 * x) for x in rgb] for rgb in RGBs]
    return RGBs