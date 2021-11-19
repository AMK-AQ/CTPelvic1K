# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:56:34 2021

@author: akh
"""



from batchgenerators.utilities.file_and_folder_operations import *

def remove_trailing_slash(filename:str):
    return os.path.normpath(filename)
    
def get_last_folder(foldername: str):
    return os.path.basename(remove_trailing_slash(folder))
    
    
def maybe_add_0000_to_all_niigz(folder):
    nii_gz = subfiles(folder, suffix='.nii.gz')
    for n in nii_gz:
        n = remove_trailing_slash(n)
        if not n.endswith('_0000.nii.gz'):
            os.rename(n, n[:-7] + '_0000.nii.gz')
            
            