""" the code are adapted from GANet CVPR2019 Paper code:
    > see the code at: https://github.com/feihuzhang/GANet/tree/ab6782aff8b21cf2a70f3f839fc4030d24b29a1c/dataloader
"""

#from .dataset import DatasetFromList
from .dataset import IterableDatasetFromList, DatasetFromList

from PIL import Image
import numpy as np

def get_iter_testing_set(
        data_path, 
        test_list, 
        crop_size=[256,256], 
        kitti2012=False, 
        kitti2015=False, 
        eth3d = False,
        middlebury = False,
        maxdisp=192,
        is_semantic=True,
        args_dict = None,
        is_left_only = True,
        encoder_ds = 64 # e.g., == 64, encoder sacle;
        ):
    return IterableDatasetFromList(
            data_path = data_path, 
            file_list_txt = test_list,
            crop_size = crop_size, 
            training = False,
            kitti2012 = kitti2012, 
            kitti2015 = kitti2015, 
            eth3d = eth3d,
            middlebury = middlebury,
            maxdisp = maxdisp, 
            is_semantic = is_semantic,
            args_dict = args_dict,
            is_left_only = is_left_only,
            encoder_ds = encoder_ds # e.g., == 64, encoder sacle;
            )

def get_iter_training_set(data_path, 
        train_list, 
        crop_size=[256,256], 
        #left_right=False, 
        kitti2012=False, 
        kitti2015=False, 
        eth3d = False,
        middlebury = False,
        maxdisp=192,
        is_semantic=True,
        args_dict = None,
        is_left_only = True,
        ):
    
    return IterableDatasetFromList(
            data_path = data_path, 
            file_list_txt = train_list,
            crop_size = crop_size, 
            training = True,
            kitti2012 = kitti2012, 
            kitti2015 = kitti2015, 
            eth3d = eth3d,
            middlebury = middlebury,
            maxdisp = maxdisp, 
            is_semantic = is_semantic,
            args_dict = args_dict,
            is_left_only = is_left_only
            )

def get_training_set(data_path,
        train_list, 
        crop_size=[256,256], 
        #left_right=False, 
        kitti2012=False, 
        kitti2015=False, 
        eth3d = False,
        middlebury = False,
        maxdisp=192,
        is_semantic=True,
        args_dict = None,
        is_left_only = True,
        ):
    
    return DatasetFromList(
            data_path = data_path, 
            file_list_txt = train_list,
            crop_size = crop_size, 
            training = True,
            kitti2012 = kitti2012, 
            kitti2015 = kitti2015, 
            eth3d = eth3d,
            middlebury = middlebury,
            maxdisp = maxdisp, 
            is_semantic = is_semantic,
            args_dict = args_dict,
            is_left_only = is_left_only
            )
