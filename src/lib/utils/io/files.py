""" Module utils/io/files.py 
Contains common operations on creating, checking, listing files/dirs.

API is organized in 3 sections:
1. Post - creation of directories, files
2. Get - query of file types, listing directories or certain files subject
     to name constraints.
"""

import sys, os
import shutil
import pathlib
import ntpath

### Constants ###
IMAGE_EXTS = ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif',
              '.nii', '.nii.gz', '.dcm']


### ======================================================================== ###
### * ### * ### * ### *          API Definitions         * ### * ### * ### * ### 
### ======================================================================== ###

__all__ = ['create_dirs_from_file',
           'is_image', 
           'list_images', 'list_files', 'list_dirs', 'list_all_files_recursive',
           'get_ext', 'get_filename',
           'natural_sort']

### ---- ### ---- \\    POST     // ---- ### ---- ###

def create_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_dirs_from_file(filepath):
    r""" Creates directory structure (if any part is missing) from filepath. """
    dirs = os.path.dirname(filepath)
    if not dirs:
        return
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)


### ---- ### ---- \\    GET     // ---- ### ---- ###

def is_image(filepath, exts=IMAGE_EXTS, checkfile=False, case_sensitive=False):
    r"""
    Given a file's fullpath, return boolean if it is an image file or not.
    Params:
        filepath - full path to file
        ?exts - list of recognized image extensions (e.g. ['jpg','png'])
    """
    # Doesn't check exts validity for performance reasons
    if checkfile:
        assert os.path.isfile(filepath), f"File ({filepath}) does not exist."
    if isinstance(exts, str):
        exts = [exts]
    assert sum(list(map(lambda x: isinstance(x, str), exts))) == len(exts), \
        f"exts parameter must be a single string or list of strings"
    for ext in exts:
        file_ext = filepath[-len(ext):]
        if (case_sensitive and file_ext == ext) or \
          (not case_sensitive and file_ext.lower() == ext.lower()):
            return True
    return False


def list_images(path, exts=None, sort=True, fullpath=False):
    r"""
    Returns list of images in a given directory with custom specifications.
        Note that list_files is the generalized version of this function.
    Params:
        path - full path to directory of images (asserts existence)
        ?ext - list of custom exts to define image criteria (e.g. ['.jpg','.png])
                if left empty, uses default image extensions
        ?sort - choice to return natural sorted list or not
        ?fullpath - choice to return just image names or full paths
    """
    assert os.path.isdir(path), f"Path ({path}) is not valid."
    if exts is not None:
        assert isinstance(exts, list) or isinstance(exts, tuple)
        assert len(exts) >= 1
    else:
        exts = IMAGE_EXTS
    files_list = [f for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and 
                     is_image(os.path.join(path, f), exts=exts)]
    if sort:
        files_list = natural_sort(files_list)
    if fullpath:
        files_list = [os.path.join(path, fn) for fn in files_list]
    return files_list


def list_files(path, subnames=[], hidden=False, sort=True, fullpath=False):
    r"""
    Returns list of files in a directory path with customizable specifications.
    Params:
        path - full path to directory (asserts)
        ?subnames - single str or list of subnames to check for in a file
                    (e.g. ['.jpg','dog'] includes only dog jpg images)
        ?hidden - boolean choice to include hidden files (.*)
        ?sort - boolean choice to natural sort final list
        ?fullpath - boolean choice to return just filenames or full file paths
    """
    if os.path.isfile(path):  # if given file, returns file path with param opts
        if get_filename(path)[0] == '.' and not hidden:
            return []
        if fullpath:
            return [os.path.abspath(path)]
        return [path]
    
    assert os.path.isdir(path), f"Path ({path}) is not valid."
    if isinstance(subnames, str):
        subnames = [subnames]
    assert isinstance(subnames, list)
    assert sum(list(map(lambda x: isinstance(x, str), subnames))) == len(subnames)
    file_list = [f for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f))]
    ret_list = []
    for fn in file_list:
        if fn[0] == '.' and not hidden:
            continue
        if not subnames:  # empty list
            ret_list.append(fn)
            continue
        for subname in subnames:
            if subname in fn:
                ret_list.append(fn)
                break 
    if sort:
        ret_list = natural_sort(ret_list)
    if fullpath:
        ret_list = [os.path.abspath(os.path.join(path, fn)) for fn in ret_list]
    return ret_list


def list_dirs(path, subnames=[], hidden=False, sort=True, fullpath=False):
    r"""
    Returns list of directories in a path with customizable specifications.
    Params:
        path - full path to directory (asserts exists) or file
        ?subnames - single str or list of subnames to check for in a directory
                    (e.g. ['data'] includes only names with data in them)
        ?hidden - boolean choice to include hidden directories (.*)
        ?sort - boolean choice to natural sort final list
        ?fullpath - boolean choice to return just dirnames or full dir paths
    """
    assert os.path.isdir(path), f"Path ({path}) is not valid or not a directory."
    if isinstance(subnames, str):
        subnames = [subnames]
    assert isinstance(subnames, list)
    assert sum(list(map(lambda x: isinstance(x, str), subnames))) == len(subnames)
    dir_list = [d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))]
    ret_list = []
    for dn in dir_list:
        if dn[0] == '.' and not hidden:
            continue
        if not subnames:  # empty list
            ret_list.append(dn)
            continue
        for subname in subnames:
            if subname in dn:
                ret_list.append(dn)
                break 
    if sort:
        ret_list = natural_sort(ret_list)
    if fullpath:
        ret_list = [os.path.join(path, dn) for dn in ret_list]
    return ret_list


def list_all_files_recursive(dirpaths, subnames=[], 
                             hidden=False, sort=True, fullpath=True):
    r"""
    Recursively traverses all directories and lists files.
    """
    if isinstance(dirpaths, list) or isinstance(dirpaths, tuple):
        assert sum([os.path.isdir(t) for t in dirpaths]) == len(dirpaths)
    elif isinstance(dirpaths, str):
        assert os.path.isdir(dirpaths), \
               f"Path ({dirpaths}) is not valid or not a directory."
        dirpaths = [dirpaths]
    else:
        raise TypeError(f"Type of dirpaths ({type(dirpaths)}) is invalid.")
    
    files = []
    for dp in dirpaths:
        files += list_files(dp, subnames=subnames, hidden=hidden, sort=sort,
                            fullpath=fullpath)
        more_dirs = list_dirs(dp, hidden=hidden, fullpath=True)
        for mdp in more_dirs:
            files += list_all_files_recursive(mdp, subnames=subnames)

    return files


def get_ext(path, dot=True):
    r"""
    Returns file extension given a single filename or its full path.
    Params:
        path - full path to file (does not check if file actually exists)
        ?dot - choice of whether to return extension with dot in front or not
    """
    cleaned_fn = get_filename(path, ext=True)
    _, dot_ext = os.path.splitext(cleaned_fn)
    if dot:
        return dot_ext
    return dot_ext[1:]


def get_filename(path, ext=True):
    r"""
    Returns file name given a single filename or its full path.
    Params:
        path - full path to file (does not check if file actually exists)
        ?ext - choice of whether to return filename with an extension or not
    """
    # with paths ending in '/', removes that and assigns to head with tail=None
    head, tail = ntpath.split(path)  
    fn = tail or ntpath.basename(head)
    if ext:
        return fn
    return fn[:len(fn)-len(get_ext(fn, dot=True))]


### ---- ### ---- \\    DELETE     // ---- ### ---- ###


def delete_file(filepath):
    r""" Deletes a file if it exists. """
    if os.path.isfile(filepath):
        os.remove(filepath)


### ---- ### ---- \\    Other Utilities     // ---- ### ---- ###

def natural_sort(name_list):
    r"""
    Returns new, sorted list of name_list.
    """
    assert isinstance(name_list, list)
    if len(name_list) == 0:
        return []
    assert sum(list(map(lambda x: isinstance(x, str), name_list))) == len(name_list)
    
    from re import compile, split
    dre = compile(r'(\d+)')
    ret_list = sorted(name_list, 
                      key=lambda l: [int(s) if s.isdigit() else s.lower() 
                                     for s in split(dre, l)])
    return ret_list



### ======================================================================== ###
### * ### * ### * ### *         Rudimentary Tests        * ### * ### * ### * ### 
### ======================================================================== ###    


if __name__ == '__main__':
    
    # TEST is_image
    print(f"\n--------\nTesting list_files..\n")
    img = '/subset_train/0_0/1.png'
    # print(f"img ({img}) --> {is_image(img, exts=['jpg'])}")
    img = '/subset_train/0_0/1.png'
    print(f"img ({img}) --> {is_image(img, exts=[])}")
    img = '/subset_train/0_0/1.png'
    print(f"img ({img}) --> {is_image(img, exts=['.png'])}")

    # TEST list_files
    print(f"\n--------\nTesting list_files..\n")
    ex_dir = '/Users/charzhar/Desktop/[Project] rere/rere/datasets/data/mnist/subset_train/0_0'
    print(list_files('/Users/charzhar/Desktop/[Project] rere/rere/datasets/data/others'))
    print(list_files(ex_dir, subnames=['tst']))
    print(list_files(ex_dir, subnames=['111','2222']))
    print(list_files(ex_dir, subnames=['599'], fullpath=True)[:5], '...')

    # TEST list_dirs
    print(f"\n--------\nTesting list_dirs..\n")
    ex_dir = '/Users/charzhar/Desktop/[Project] rere/rere/datasets/data/mnist/subset_train'
    print(list_dirs('/Users/charzhar/Desktop/[Project] rere/rere/datasets/data/mnist/subset_train/0_0'))
    print(list_dirs(ex_dir, subnames=['A','1','2']))
    print(list_dirs(ex_dir, subnames=['A','1','2','_']))
    print(list_dirs(ex_dir, subnames=['1','2','3'], fullpath=True))

    # TEST get_filename and get_ext
    print()
    fn = 'mn/ist01.png'
    print(f"ext for ({fn}) -> ({get_ext(fn)})")
    fn = 'mn.ist01.png'
    print(f"ext for ({fn}) -> ({get_ext(fn)})")
    fn = '/desktop/mnistnotes'
    print(f"ext for ({fn}) -> ({get_ext(fn)})")
    fn = '/desktop/mnistnotes.'
    print(f"ext for ({fn}) -> ({get_ext(fn)})")
    fn = '/desktop/mnistnotes./'
    print(f"ext for ({fn}) -> ({get_ext(fn)})")
    fn = '/desktop/mn.ist01.png/'
    print(f"ext for ({fn}) -> ({get_ext(fn)})")
    fn = '/desktop/.mn.ist01.png'
    print(f"ext for ({fn}) -> ({get_ext(fn, dot=False)})")
    
    print()
    fn = 'mnist01.png'
    print(f"filename for ({fn}) -> ({get_filename(fn)})")
    fn = 'mnist01.png/'
    print(f"filename for ({fn}) -> ({get_filename(fn)})")
    fn = '/mnist01.png'
    print(f"filename for ({fn}) -> ({get_filename(fn)})")
    fn = '/mnist01.png/'
    print(f"filename for ({fn}) -> ({get_filename(fn)})")
    fn = 'desktop/mnist/mnist01.png/'
    print(f"filename for ({fn}) -> ({get_filename(fn, ext=False)})")
    fn = 'desktop/mnist/mnistnotes/'
    print(f"filename for ({fn}) -> ({get_filename(fn, ext=False)})")

