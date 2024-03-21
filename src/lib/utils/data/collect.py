
import math
import logging
import pathlib
import re
import pandas as pd


# ------ Image File Access, Organizing, and Sorting ----- #

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# ------ Dataframe Access & Manipulation ----- #

def correct_df_directories(df, path):
    """
    Assumptions:
        - Hard-coded to check 'image' & 'mask' keys.
        - All files in a df are from the same main dataset directory.
            i.e. Single Dataset
        - The basic file structures are the same in the dataset directory.
    """
    # check if replacement is needed
    path = pathlib.Path(path)
    img_path = pathlib.Path(df.iloc[0]['image'])  # reference
    if path in img_path.parents:
        return df

    # Get number of path elements to keep relative to the root of dataset
    logging.info(f'correct_df_directories(): path adjustment necessary..')
    indices_to_keep = 0
    for p in img_path.parents:
        indices_to_keep += 1
        if path.name == p.name:
            break
    assert indices_to_keep != 0, 'Images should not be in dataset base dir.'

    def get_new_path(old_image_path, new_base_path, len_relative_path):
        if pd.isna(old_image_path) or not old_image_path:
            return old_image_path
        old_image_path = pathlib.PurePath(old_image_path)
        relative_image_path = old_image_path.parts[-len_relative_path:]
        new_image_path = new_base_path.joinpath(*relative_image_path)
        assert new_image_path.exists(), f'File {new_image_path} doesn\'t exist'
        return str(new_image_path)

    for i, S in df.iterrows():
        df.at[i, 'image'] = get_new_path(S['image'], path, indices_to_keep)
        df.at[i, 'mask'] = get_new_path(S['mask'], path, indices_to_keep)
    return df


