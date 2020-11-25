"""
# the code is adapted from GANet paper code:
# > see the code at: https://github.com/feihuzhang/GANet/tree/ab6782aff8b21cf2a70f3f839fc4030d24b29a1c/dataloader
"""
import torch.utils.data as data
#import skimage
#import skimage.io
#import skimage.transform

from PIL import Image
import numpy as np
import random
from struct import unpack
import re
import sys


from os.path import join as pjoin
import src.pfmutil as pfm
import cv2
from .cbmv_generator import (
        generate_crop_train_cbmv, 
        get_default_args_dict, 
        generate_dummy_crop_train_cbmv,
        generate_test_cbmv
        )
import torch

# load sf (scene flow) data
def load_sfdata(data_path, current_file, is_cleanpass=False):
    A = current_file
    if is_cleanpass:
        A = A.replace('frames_finalpass', #old 
                      'frames_cleanpass', #new
                      1 # count
                      )
    limg_name = pjoin(data_path, A)
    #print ("[****] limg: {}".format(limg_name[len(data_path):]))
    rimg_name = pjoin(data_path, A[:-13] + 'right/' + A[len(A)-8:])
    #print ("[****] rimg: {}".format(rimg_name))
    pos = A.find('/')
    tmp_len = len('frames_finalpass')
    ldisp_name = pjoin(data_path, A[0:pos] + '/disparity' + A[pos+1+tmp_len:-4] + '.pfm')
    #print ("[****] ldisp: {}".format(ldisp_name))
    #disp_left = pfm.readPFM(filename)
    #print ('[***] disp_left shape = ', disp_left.shape)
    #pfm.show(disp_left)
    #print ("[???] ",data_path +  'disparity/' + A[0:-13] + 'right/' + A[len(A)-8:-4] + '.pfm' )
    
    #rdisp_name = pjoin(data_path, A[0:pos] + '/disparity' + A[pos+1+tmp_len:-13] + 'right/' + A[len(A)-8:-4] + '.pfm')
    #print ("[****] rdisp: {}".format(rdisp_name))
    return limg_name, rimg_name, ldisp_name


def load_kitti2012_data(file_path, current_file):
    """ load current file from the list"""
    #limg_name = pjoin(file_path, 'colored_0/' + current_file)
    limg_name = pjoin(file_path, 'image_0/' + current_file)
    #print ("limg: {}".format(limg_name))
    rimg_name = pjoin(file_path, 'image_1/' + current_file)
    #rimg_name = pjoin(file_path, 'colored_1/' + current_file)
    #print ("rimg: {}".format(rimg_name))
    
    ldisp_name = pjoin(file_path, 'disp_occ_pfm/' + current_file[0:-4]+ '.pfm')
    #print ("ldisp: {}".format(ldisp_name))
    return limg_name, rimg_name, ldisp_name



def load_eth3d_data(file_path, current_file):
    """ load current file from the list"""
    limg_name = pjoin(file_path, current_file + '/im0.png')
    #print ("limg: {}".format(limg_name))
    rimg_name = pjoin(file_path, current_file + '/im1.png')
    #print ("rimg: {}".format(rimg_name))
    
    ldisp_name = pjoin(file_path, current_file + '/disp0GT.pfm')
    #print ("ldisp: {}".format(ldisp_name))
    return limg_name, rimg_name, ldisp_name

def load_middlebury_data(file_path, current_file):
    """ load current file from the list"""
    limg_name = pjoin(file_path, current_file + '/im0.png')
    #print ("limg: {}".format(limg_name))
    rimg_name = pjoin(file_path, current_file + '/im1.png')
    #print ("rimg: {}".format(rimg_name))
    
    ldisp_name = pjoin(file_path, current_file + '/disp0GT.pfm')
    #print ("ldisp: {}".format(ldisp_name))
    return limg_name, rimg_name, ldisp_name

def load_kitti2015_data(file_path, current_file, is_semantic = True):
    """ load current file from the list"""
    limg_name  = pjoin(file_path, 'image_0/' + current_file)
    #print ("limg: {}".format(limg_name ))
    
    rimg_name  = pjoin(file_path, 'image_1/' + current_file)
    #print ("rimg: {}".format(limg_name ))
    #right = np.asarray(Image.open(filename), dtype=np.float32, order="C")
    ldisp_name = file_path + 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm'
    #print ("ldisp: {}".format(ldisp_name))
    #disp_left = pfm.readPFM(filename)
    # semantic segmentaion label
    if is_semantic:
        # uint8 gray png image
        filename = pjoin(file_path, '../data_semantics/training/semantic/' + current_file)
        lseg_name = filename
        #print ("semantic label: {}".format(filename))
        #semantic_label = np.asarray(Image.open(filename), dtype=np.float32, order="C")
        #pfm.show(semantic_label)
        #temp_data[14,:,:] = semantic_label
    else:
        lseg_name = None
    return limg_name, rimg_name, ldisp_name, lseg_name


# added by CCJ:
# NOTE: ??? Problem: Dataloader with num_workers>=1 stuck at 
# epoch 1 (i.e., cannot start next epoch automatically)
# when loading large dataset for CBMV feature extraction, 
# e.g., loading SF dataset with 30k images ,

# Dataset: assume that you can trivially map each data point in your dataset, e.g, image feeding;
class DatasetFromList(data.Dataset): 
    def __init__(self, data_path, file_list_txt, crop_size=[256, 256], 
            training=True, 
            
            kitti2012=False, 
            kitti2015=False, 
            eth3d = False,
            middlebury = False,

            maxdisp=192, 
            is_semantic=True, 
            args_dict = None,
            is_left_only= True
            ):
        super(DatasetFromList, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        f = open(file_list_txt, 'r')
        self.data_path = data_path
        self.file_list = [l.rstrip() for l in f.readlines()]
        print ("[***] img# = {}, file_list = {}, ..., {}".format(len(self.file_list), 
            self.file_list[0], self.file_list[len(self.file_list) - 1]))
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.kitti2012 = kitti2012
        self.kitti2015 = kitti2015
        self.eth3d = eth3d
        self.middlebury = middlebury,
        self.maxdisp = maxdisp
        self.is_semantic = is_semantic
        self.is_left_only = is_left_only
        
        if any ([self.kitti2012, self.eth3d, self.middlebury]):
            self.is_semantic = False
        if args_dict is None:
            self.args_dict = get_default_args_dict()
        else:
            self.args_dict = args_dict
       
        self.is_cleanpass = args_dict['sf_frames_type'] == 'frames_cleanpass'
        print ('[****] Using SF ', args_dict['sf_frames_type'])

    def __getitem__(self, index):
    #    print self.file_list[index]
        if self.kitti2012: #load kitti2012 dataset
            temp_data = load_kitti2012_data(self.data_path, self.file_list[index])
        elif self.kitti2015: #load kitti2015 dataset
            temp_data = load_kitti2015_data(self.data_path, self.file_list[index], self.is_semantic)
        elif self.eth3d: #load eth3d dataset
            temp_data = load_eth3d_data(self.data_path, self.file_list[index])
        elif self.middlebury: #load middlebury dataset
            temp_data = load_middlebury_data(self.data_path, self.file_list[index])
        else: #load scene flow dataset
            #assert index < len(self.file_list), 'index=%d, len=%d'%(index, len(self.file_list))
            temp_data = load_sfdata(self.data_path, self.file_list[index], self.is_cleanpass)
        
        if self.training:
            is_fixed_center_around_crop = False
        else:
            is_fixed_center_around_crop = True
        """
        cbmv_feature, disp, imgl_rgb, semantic_label = generate_dummy_crop_train_cbmv(
            limg_name = temp_data[0], # left image name
            rimg_name = temp_data[1], # right image name
            disp_name = temp_data[2], # left disparity file name
            lseg_name = temp_data[3] if self.is_semantic else None,  # left segmentation GT file name
            crop_height = self.crop_height,
            crop_width = self.crop_width,
            maxdisp = self.maxdisp,
            is_fixed_center_around_crop = is_fixed_center_around_crop,
            args_dict = self.args_dict,
            is_left_only= self.is_left_only
            )
        """
        
        cbmv_feature, disp, imgl_rgb, imgr_rgb, semantic_label = generate_crop_train_cbmv(
            limg_name = temp_data[0], # left image name
            rimg_name = temp_data[1], # right image name
            disp_name = temp_data[2], # left disparity file name
            lseg_name = temp_data[3] if self.is_semantic else None,  # left segmentation GT file name
            crop_height = self.crop_height,
            crop_width = self.crop_width,
            maxdisp = self.maxdisp,
            is_fixed_center_around_crop = is_fixed_center_around_crop,
            args_dict = self.args_dict,
            is_left_only= self.is_left_only
            )
        
        return cbmv_feature, disp, imgl_rgb, imgr_rgb, semantic_label

    def __len__(self):
        return len(self.file_list)


# added by CCJ:
# IterableDataset: helps to large data stream, like, an audio or video feed.
class IterableDatasetFromList(data.IterableDataset): 
    r"""An iterable Dataset.

        All datasets that represent an iterable of data samples should subclass it.
        Such form of datasets is particularly useful when data come from a stream.

        All subclasses should overwrite :meth:`__iter__`, which would return an
        iterator of samples in this dataset.
    """
    def __init__(self, data_path, file_list_txt, crop_size=[256, 256], 
            training=True, 
            kitti2012=False, 
            kitti2015=False, 
            eth3d = False,
            middlebury = False,
            maxdisp=192, 
            is_semantic=True, 
            is_shuffle = True, # set to True to have the data reshuffled at every epoch
            args_dict = None,
            is_left_only= True,
            encoder_ds = 64 # e.g., == 64, encoder sacle;
            ):
        super(IterableDatasetFromList, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        f = open(file_list_txt, 'r')
        self.data_path = data_path
        self.file_list = [l.rstrip() for l in f.readlines()]
        print ("[***] Iterable Dataset!!! Img# = {}, file_list = {}, ..., {}".format(len(self.file_list), 
            self.file_list[0], self.file_list[len(self.file_list) - 1]))
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.kitti2012 = kitti2012
        self.kitti2015 = kitti2015
        self.eth3d = eth3d
        self.middlebury = middlebury
        self.maxdisp = maxdisp
        self.is_semantic = is_semantic
        self.is_left_only = is_left_only
        self.ds = encoder_ds
        if any ([self.kitti2012, self.eth3d, self.middlebury]):
            self.is_semantic = False
        if args_dict is None:
            self.args_dict = get_default_args_dict()
        else:
            self.args_dict = args_dict
        
        self.is_cleanpass = args_dict['sf_frames_type'] == 'frames_cleanpass'
        print ('[****] Using SF ', args_dict['sf_frames_type'])
        #print ('[****] input data:', self.kitti2012, self.kitti2015, self.eth3d, self.middlebury)

    
    def random_shuffle(self):
        print ("random shuffle for next epoch training!!!")
        n = len(self.file_list)
        print ("[***] Before shuffle: file_list = {}, ..., {}".format( self.file_list[0], 
                self.file_list[n-1]))
        self.file_list = random.sample(self.file_list, n)
        print ("[***] After shuffle: file_list = {}, ..., {}".format( self.file_list[0], 
                self.file_list[n-1]))

    def get_my_array_stream(self):
        for img_itr in self.file_list:
            if self.kitti2012: #load kitti2012 dataset
                temp_data = load_kitti2012_data(self.data_path, img_itr)
            elif self.kitti2015: #load kitti2015 dataset
                temp_data = load_kitti2015_data(self.data_path, img_itr, self.is_semantic)
            elif self.eth3d: #load eth3d dataset
                temp_data = load_eth3d_data(self.data_path, img_itr)
            elif self.middlebury: #load middlebury dataset
                temp_data = load_middlebury_data(self.data_path, img_itr)
            else: #load scene flow dataset
                temp_data = load_sfdata(self.data_path, img_itr, self.is_cleanpass)
            
            if self.training:
                is_fixed_center_around_crop = False
            else:
                is_fixed_center_around_crop = True
            """
            #dummy one
            cbmv_feature, disp, imgl_rgb, semantic_label = generate_dummy_crop_train_cbmv(
                limg_name = temp_data[0], # left image name
                rimg_name = temp_data[1], # right image name
                disp_name = temp_data[2], # left disparity file name
                lseg_name = temp_data[3] if self.is_semantic else None,  # left segmentation GT file name
                crop_height = self.crop_height,
                crop_width = self.crop_width,
                maxdisp = self.maxdisp,
                is_fixed_center_around_crop = is_fixed_center_around_crop,
                args_dict = self.args_dict,
                is_left_only= self.is_left_only
                )

            """
            if self.training: 
                cbmv_feature, disp, imgl_rgb, imgr_rgb, semantic_label = generate_crop_train_cbmv(
                    limg_name = temp_data[0], # left image name
                    rimg_name = temp_data[1], # right image name
                    disp_name = temp_data[2], # left disparity file name
                    lseg_name = temp_data[3] if self.is_semantic else None,  # left segmentation GT file name
                    crop_height = self.crop_height,
                    crop_width = self.crop_width,
                    maxdisp = self.maxdisp,
                    is_fixed_center_around_crop = is_fixed_center_around_crop,
                    args_dict = self.args_dict,
                    is_left_only = self.is_left_only
                    )
                yield cbmv_feature, disp, imgl_rgb, imgr_rgb, semantic_label
            else:
                disp_name = temp_data[2] # left disparity file name
                cbmv_feature, height, width, crop_height, crop_width = generate_test_cbmv(
                    limg_name = temp_data[0], 
                    rimg_name = temp_data[1], 
                    crop_height = self.crop_height, 
                    crop_width = self.crop_width,
                    encoder_ds = self.ds,
                    maxdisp = self.maxdisp,
                    args_dict = self.args_dict,
                    # only generate feature for left image; otherwise, for both left and right;
                    is_left_only = self.is_left_only 
                    )
                current_file = img_itr
                yield cbmv_feature, height, width, crop_height, crop_width, current_file, disp_name

    
    def __iter__(self):
        return self.get_my_array_stream()

# Define a `worker_init_fn` that configures each dataset copy differently
def my_worker_init_fn(unsed_arg):
    #import math
    worker_info = torch.utils.data.get_worker_info()
    cur_dataset = worker_info.dataset  # the dataset copy in this worker process
    worker_id = worker_info.id
    # configure the dataset to only process the split workload
    N = len(cur_dataset.file_list)
    split_size = N // worker_info.num_workers
    cur_dataset.file_list = cur_dataset.file_list[worker_id*split_size : min(N,(worker_id+1)*split_size)]

# Mult-process loading with the custom `worker_init_fn`
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
#print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
# >>> [3, 5, 4, 6]

# With even more workers
# >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
# >>> [3, 4, 5, 6]
