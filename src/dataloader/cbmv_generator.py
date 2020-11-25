import cv2
import os
import numpy as np
import glob
import random
from datetime import datetime
from .. import pfmutil as pfm
import math
import sys
import scipy.misc
import torch

#*********************
# matcher wrapper
#*********************
import src.cpp.lib.libmatchers as mtc
import src.cpp.lib.libfeatextract as fte
import skimage
from PIL import Image
#from termcolor import colored

from ..funcs_utili import show_2_imgs_1_row, show_3_imgs_1_row, show_4_imgs_3_row,show_4_imgs_2_row, show_4_imgs_4_row
from .. import pfmutil as pfm

""" cost for left image """
def get_costs(iml,imr, maxdisp = 192, censw = 11, nccw = 3, sadw = 5, sobelw = 5, board_h = 10, board_w_left = 10, board_w_right = 0):
        #costcensus shape : [img_H, img_W, ndisp] 
        costcensus = mtc.census(iml,imr, maxdisp,censw).astype(np.float32)
        #print ("[???]", costcensus.shape)
        
        #cost_tmp = costcensus
        #cost_tmp[np.isclose(cost_tmp, 2147483648.0)] = 0.1
        #print 'costcensus max = {}, min = {}'.format(np.max(cost_tmp), np.min(cost_tmp))
        #pfm.show(iml)
        #pfm.show(imr)
        #disp_census = np.argmin(costcensus, axis = 2).astype(np.float32)              
        #pfm.show(np.argmin(costcensus, axis = 2).astype(np.float32))
        #sys.exit()

        #costncc shape : [ndisp, img_H, img_W]
        costncc = mtc.nccNister(iml,imr,maxdisp,nccw).astype(np.float32)
        #after fte.swap_axes(), now costncc shape : [img_H, img_W, ndisp]
        costncc = fte.swap_axes(costncc)
        
        #cost_tmp = costncc
        #cost_tmp[np.isclose(cost_tmp, 2147483648.0)] = 0.1
        #print '[???]costncc max = {}, min = {}'.format(np.max(cost_tmp), np.min(cost_tmp))

        costsad = mtc.zsad(iml,imr, maxdisp, sadw).astype(np.float32)
        #costsad shape : [img_H, img_W, ndisp]
        costsad = fte.swap_axes(costsad)
        
        #cost_tmp = costsad
        #cost_tmp[np.isclose(cost_tmp, 2147483648.0)] = 0.1
        #print 'costsad max = {}, min = {}'.format(np.max(cost_tmp), np.min(cost_tmp))
        
        sobl = mtc.sobel(iml)
        sobr = mtc.sobel(imr)
        costsob = mtc.sadsob(sobl,sobr, maxdisp, sobelw).astype(np.float32)
        #costsob shape : [img_H, img_W, ndisp]
        costsob = fte.swap_axes(costsob)
        #cost_tmp = costsob
        #cost_tmp[np.isclose(cost_tmp, 2147483648.0)] = 0.1
        #print 'costsob max = {}, min = {}'.format(np.max(cost_tmp), np.min(cost_tmp))

        #pfm.show(np.argmin(costsob, axis = 2).astype(np.float32))
        #pfm.save('results/ncc.pfm', np.argmin(costncc, axis = 2).astype(np.float32))
        #sys.exit()

        # add max(board_w_right,  1), due to the case of board_w_right = 0;
        #valid ending index for width;
        vld_w_end = - board_w_right if board_w_right > 0 else None
        # valid ending index for height;
        vld_h_end = - board_h if board_h > 0 else None
        return costcensus[board_h:vld_h_end,board_w_left:vld_w_end,:].copy(order='C'), \
               costncc[board_h:vld_h_end,board_w_left:vld_w_end,:].copy(order='C'), \
               costsob[board_h:vld_h_end,board_w_left:vld_w_end,:].copy(order='C'), \
               costsad[board_h:vld_h_end,board_w_left:vld_w_end,:].copy(order='C')


#NOTE: added by CCJ for iResNet, due to feature constancy between left cbmv features and right cbmv features;
# extract left and right features for efficient feature computation;
def extract_features_lr(
    census,ncc,sobel,sad, 
    cens_sigma = 128.0,  
    ncc_sigma=0.02, 
    sad_sigma=20000.0, 
    sobel_sigma=20000.0,
    disp_image = None):

        # cost shape : [img_H, img_W, ndisp]
        h,w,ndisp = census.shape[:]
        censusR =  fte.get_right_cost(census)
        #pfm.show(np.argmin(census, axis = 2).astype(np.float32), title='disp_census')
        #pfm.show(np.argmin(censusR, axis = 2).astype(np.float32), title='R_disp_census')

        nccR = fte.get_right_cost(ncc)
        #pfm.show(np.argmin(ncc, axis = 2).astype(np.float32), title='disp_ncc')
        #pfm.show(np.argmin(nccR, axis = 2).astype(np.float32), title='R_disp_ncc')

        sadR = fte.get_right_cost(sad)
        #pfm.show(np.argmin(sad, axis = 2).astype(np.float32), title='disp_sad')
        #pfm.show(np.argmin(sadR, axis = 2).astype(np.float32), title='R_disp_sad')
        
        sobelR = fte.get_right_cost(sobel)
        #pfm.show(np.argmin(sobel, axis = 2).astype(np.float32), title='disp_sob')
        #pfm.show(np.argmin(sobelR, axis = 2).astype(np.float32), title='R_disp_sob')
        
        #cost_tmp = census
        #cost_tmp[np.isclose(cost_tmp, 2147483648.0)] = 1024.0
        census  =  np.reshape(census,   [h*w, ndisp])
        ncc  =  np.reshape(ncc, [h*w,ndisp])
        sobel  =  np.reshape(sobel, [h*w,ndisp])
        sad  =  np.reshape(sad,  [h*w,ndisp])
        
        censusR =  np.reshape(censusR,  [h*w, ndisp])
        nccR =  np.reshape(nccR, [h*w, ndisp])
        sobelR =  np.reshape(sobelR,[h*w,ndisp])
        sadR =  np.reshape(sadR, [h*w,ndisp])
        
        """
        # just for debugging!!! 
        # left ones!
        imgs_to_plot = []
        d0 = 25
        tmp_disp1 = disp_image.copy(order = 'C')
        tmp_disp2 = disp_image.copy(order = 'C')
        tmp_disp3 = disp_image.copy(order = 'C')
        tmp_disp1[np.abs(tmp_disp1 - d0) > 3] = 0
        tmp_disp2[np.abs(tmp_disp2 - d0) > 2] = 0
        tmp_disp3[np.abs(tmp_disp3 - d0) > 1] = 0
        # 0, 1, 2, 3;
        imgs_to_plot.append(disp_image)
        imgs_to_plot.append(tmp_disp1)
        imgs_to_plot.append(tmp_disp2)
        imgs_to_plot.append(tmp_disp3)
        # 4, 5, 6, 7;
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(census, cens_sigma/ (120.0**2 *4)),[h,w,ndisp])[:,:,d0], [h,w]))
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(ncc,    ncc_sigma / 4.0),[h,w,ndisp])[:,:,d0], [h,w]))
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(sobel,  sobel_sigma / float (2**26*4)),[h,w,ndisp])[:,:,d0], [h,w]))
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(sad,    sad_sigma / float (2**26*16)),[h,w,ndisp])[:,:,d0], [h,w]))
        
        # 8, 9, 10, 11;
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(census, cens_sigma/ (120.0**2)),[h,w,ndisp])[:,:,d0], [h,w]))
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(ncc,    ncc_sigma/ 1.0),[h,w,ndisp])[:,:,d0], [h,w]))
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(sobel,  sobel_sigma/ float(2**26)),[h,w,ndisp])[:,:,d0], [h,w]))
        imgs_to_plot.append(np.reshape(np.reshape(fte.extract_likelihood(sad,    sad_sigma/ float(2**26)),[h,w,ndisp])[:,:,d0], [h,w]))

        features_name = [
                'censusL', 'nccL', 'sobelL', 'sadL',
                'ratio_cenL', 'ratio_nccL', 'ratio_sobL', 'ratio_sadL',
                'likly_cenL', 'likly_nccL', 'likly_sobL', 'likly_sadL',
                'ratio_cenR', 'ratio_nccR', 'ratio_sobR', 'ratio_sadR',
                'likly_cenR', 'likly_nccR', 'likly_sobR', 'likly_sadR',
                ]
        show_4_imgs_3_row(imgs = imgs_to_plot, 
                img_names = ['slice_{}_d{}'.format(i, d0) for i in features_name[0:12]],
                #cmap = ['inferno']*8 + ['gray'] * 4
                #cmap = ['inferno']*12
                cmap = ['gray']*12
                )
        
        # right ones
        imgs_to_plot = []
        features_name = [
                'disp_censusL', 'disp_nccL', 'disp_sobelL', 'disp_sadL',
                'disp_likly_cenL', 'disp_likly_nccL', 'disp_likly_sobL', 'disp_likly_sadL',
                'disp_censusR', 'disp_nccR', 'disp_sobelR', 'disp_sadR',
                'disp_likly_cenR', 'disp_likly_nccR', 'disp_likly_sobR', 'disp_likly_sadR',
                ]
        imgs_to_plot.append(np.argmin(np.reshape(census, [h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmin(np.reshape(ncc, [h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmin(np.reshape(sobel, [h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmin(np.reshape(sad, [h,w,ndisp]), axis=2))

        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(census, cens_sigma),[h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(ncc, cens_sigma),[h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(sobel, cens_sigma),[h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(sad, cens_sigma),[h,w,ndisp]), axis=2))


        imgs_to_plot.append(np.argmin(np.reshape(censusR, [h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmin(np.reshape(nccR, [h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmin(np.reshape(sobelR, [h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmin(np.reshape(sadR, [h,w,ndisp]), axis=2))
        
        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(censusR, cens_sigma),[h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(nccR, cens_sigma),[h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(sobelR, cens_sigma),[h,w,ndisp]), axis=2))
        imgs_to_plot.append(np.argmax(np.reshape(fte.extract_likelihood(sadR, cens_sigma),[h,w,ndisp]), axis=2))
        
        show_4_imgs_4_row(imgs = imgs_to_plot, 
                img_names = features_name,
                cmap = ['gray']*16
                )
        sys.exit()
        """         



        """ let channel number = 8, as the first dim, which makes 
            the following assignment and element access 
            in C contiguous arraya way, that is, a chunk of contiguous 
            memory in the C-like index order;
        """
        features = np.empty((16, h, w, ndisp), order= 'C')
        """ matcher cost : census , ncc, sobel, sad """
        ## clip to remove RAND_MAX values, and normalized to [0,1]
        features[0, :,:,:]= np.reshape(np.clip(census, 0., 120.)/120. , [h, w, ndisp])
        features[1, :,:,:]= np.reshape((1 + np.clip(ncc, -1., 1.))/2,[h, w, ndisp]) # Updated: change NCC to [0, 1] range!!!
        features[2, :,:,:]= np.reshape(np.clip(sobel, 0., 2**13)/float(2**13), [h, w, ndisp])
        features[3, :,:,:]= np.reshape(np.clip(sad, 0., 2**13) / float(2**13), [h, w, ndisp])
        """ aml for left costs: census , ncc, sobel, sad """
        # X ~ N(u, sigma^2), then aX ~ N(au, a^2*sigma^2);
        # E(aX + b) = aE(x) + b;
        # Var(aX + b) = a^2*Var(x);
        #normlized_cens_var = cens_sigma / (120.0**2 *4)
        #normalized_ncc_var = ncc_sigma/4.0
        #normalized_sob_var = sad_sigma / float(2**26 *4)
        #normalized_sad_var = sad_sigma / float(2**26 *16)
        normlized_cens_var = cens_sigma
        normalized_ncc_var = ncc_sigma
        normalized_sob_var = sad_sigma
        normalized_sad_var = sad_sigma
        
        features[4,:,:,:]=np.reshape(fte.extract_likelihood(census, normlized_cens_var),[h,w,ndisp])
        features[5,:,:,:]=np.reshape(fte.extract_likelihood(ncc, normalized_ncc_var),   [h,w,ndisp])
        features[6,:,:,:]=np.reshape(fte.extract_likelihood(sobel, normalized_sob_var), [h,w,ndisp])
        features[7,:,:,:]=np.reshape(fte.extract_likelihood(sad, normalized_sad_var),   [h,w,ndisp])


        ## Right features
        features[8, :,:,:]= np.reshape(np.clip(censusR, 0., 120.)/120. , [h, w, ndisp])
        features[9, :,:,:]= np.reshape( (1+np.clip(nccR, -1., 1.))/2,[h, w, ndisp]) # Updated: change NCC to [0, 1] range!!!
        features[10, :,:,:]= np.reshape(np.clip(sobelR, 0., 2**13)/float(2**13), [h, w, ndisp])
        features[11, :,:,:]= np.reshape(np.clip(sadR, 0., 2**13) / float(2**13), [h, w, ndisp])
        features[12,:,:,:]=np.reshape(fte.extract_likelihood(censusR, normlized_cens_var),[h,w,ndisp])
        features[13,:,:,:]=np.reshape(fte.extract_likelihood(nccR, normalized_ncc_var),   [h,w,ndisp])
        features[14,:,:,:]=np.reshape(fte.extract_likelihood(sobelR, normalized_sob_var), [h,w,ndisp])
        features[15,:,:,:]=np.reshape(fte.extract_likelihood(sadR, normalized_sad_var),   [h,w,ndisp])

        #del census
        #del ncc
        #del sobel
        #del sad
        del censusR
        del nccR
        del sobelR
        del sadR

        # in size [C, D, H, W] 
        features = features.transpose((0,3,1,2))
        return features.astype(np.float32)


# extract left features,without right-left direction for efficient feature computation;
def extract_features_left(census,ncc,sobel,sad, 
        cens_sigma = 128.0,  
        ncc_sigma=0.02, 
        sad_sigma=20000.0, 
        sobel_sigma=20000.0,
        disp_image = None):

        # cost shape : [img_H, img_W, ndisp]
        dims = census.shape
        h,w,ndisp = dims[:]
        ## clip to remove RAND_MAX values + normalize in 0,1
        #cost_tmp = census
        #cost_tmp[np.isclose(cost_tmp, 2147483648.0)] = 1024.0
        census  =  np.reshape(census,   [h*w, ndisp])
        ncc  =  np.reshape(ncc, [h*w,ndisp])
        sobel  =  np.reshape(sobel, [h*w,ndisp])
        sad  =  np.reshape(sad,  [h*w,ndisp])

        """ let channel number = 8, as the first dim, which makes 
            the following assignment and element access 
            in C contiguous arraya way, that is, a chunk of contiguous 
            memory in the C-like index order;
        """
        features = np.empty((8, h, w, ndisp), order= 'C')
        """ matcher cost : census , ncc, sobel, sad """
        features[0, :,:,:]= np.reshape(np.clip(census, 0., 120.)/120. , [h, w, ndisp])
        #features[1, :,:,:]= np.reshape(np.clip(ncc + 1.0, 0., 2.),[h, w, ndisp])
        features[1, :,:,:]= np.reshape((1 + np.clip(ncc, -1., 1.))/2,[h, w, ndisp]) # Updated: change NCC to [0, 1] range!!!
        features[2, :,:,:]= np.reshape(np.clip(sobel, 0., 2**13)/float(2**13), [h, w, ndisp])
        features[3, :,:,:]= np.reshape(np.clip(sad, 0., 2**13) / float(2**13), [h, w, ndisp])
        """ aml for left costs: census , ncc, sobel, sad """
        # X ~ N(u, sigma^2), then aX ~ N(au, a^2*sigma^2);
        # E(aX + b) = aE(x) + b;
        # Var(aX + b) = a^2*Var(x);
        #normlized_cens_var = cens_sigma / (120.0**2 *4)
        #normalized_ncc_var = ncc_sigma/4.0
        #normalized_sob_var = sad_sigma / float(2**26 *4)
        #normalized_sad_var = sad_sigma / float(2**26 *16)
        normlized_cens_var = cens_sigma
        normalized_ncc_var = ncc_sigma
        normalized_sob_var = sad_sigma
        normalized_sad_var = sad_sigma
        
        features[4,:,:,:]=np.reshape(fte.extract_likelihood(census, normlized_cens_var),[h,w,ndisp])
        features[5,:,:,:]=np.reshape(fte.extract_likelihood(ncc, normalized_ncc_var),   [h,w,ndisp])
        features[6,:,:,:]=np.reshape(fte.extract_likelihood(sobel, normalized_sob_var), [h,w,ndisp])
        features[7,:,:,:]=np.reshape(fte.extract_likelihood(sad, normalized_sad_var),   [h,w,ndisp])
        
        # in size [C, D, H, W] 
        features = features.transpose((0,3,1,2))
        return features.astype(np.float32)


# extract left features,without right-left direction for efficient feature computation;
def extract_features_left_V2(census,ncc,sobel,sad, 
        cens_sigma = 128.0,  
        ncc_sigma=0.02, 
        sad_sigma=20000.0, 
        sobel_sigma=20000.0,
        disp_image = None):

        # cost shape : [img_H, img_W, ndisp]
        dims = census.shape
        h,w,ndisp = dims[:]
        """ aml for left costs: census , ncc, sobel, sad """
        # X ~ N(u, sigma^2), then aX ~ N(au, a^2*sigma^2);
        # E(aX + b) = aE(x) + b;
        # Var(aX + b) = a^2*Var(x);
        #normlized_cens_var = cens_sigma / (120.0**2 *4)
        #normalized_ncc_var = ncc_sigma/4.0
        #normalized_sob_var = sad_sigma / float(2**26 *4)
        #normalized_sad_var = sad_sigma / float(2**26 *16)
        normlized_cens_var = cens_sigma
        normalized_ncc_var = ncc_sigma
        normalized_sob_var = sad_sigma
        normalized_sad_var = sad_sigma
        
        #census
        census_aml = np.expand_dims(np.reshape(fte.extract_likelihood(np.reshape(census, [h*w,ndisp]), normlized_cens_var),[h,w,ndisp]), 0)
        census = np.expand_dims(np.clip(census, 0., 120.)/120., 0)
        # ncc
        ncc_aml = np.expand_dims(np.reshape(fte.extract_likelihood(np.reshape(ncc,[h*w,ndisp]), normalized_ncc_var),[h,w, ndisp]), 0)
        ncc = np.expand_dims(np.clip(ncc + 1.0, 0., 2.), 0)
        # sobel
        sobel_aml = np.expand_dims(np.reshape(fte.extract_likelihood(np.reshape(sobel, [h*w,ndisp]),normalized_sob_var),[h,w,ndisp]), 0)
        sobel = np.expand_dims(np.clip(sobel, 0., 2**13) / float(2**13), 0)
        # sad
        sad_aml = np.expand_dims(np.reshape(fte.extract_likelihood(np.reshape(sad, [h*w, ndisp]), normalized_sad_var),[h,w,ndisp]), 0)
        sad = np.expand_dims(np.clip(sad, 0., 2**13) / float(2**13), 0)
        
        """ let channel number = 8, as the first dim, which makes 
            the following assignment and element access 
            in C contiguous arraya way, that is, a chunk of contiguous 
            memory in the C-like index order;
        """
        features = np.concatenate((census, ncc, sobel, sad, census_aml, ncc_aml, sobel_aml, sad_aml), 0)
        # in size [C, D, H, W] 
        features = features.transpose((0,3,1,2))
        del census_aml
        del ncc_aml
        del sobel_aml
        del sad_aml
        return features.astype(np.float32)


# input : features, in shape of [channel (e.g., = 8, or 20), ndisp, imgH, imgW];
def debug_cbmv_featues(cbmv_features):
    cbmv_features_name = [
            'censusL', 'nccL', 'sobelL', 'sadL',
            #'ratio_cenL', 'ratio_nccL', 'ratio_sobL', 'ratio_sadL',
            'likly_cenL', 'likly_nccL', 'likly_sobL', 'likly_sadL',
            #'ratio_cenR', 'ratio_nccR', 'ratio_sobR', 'ratio_sadR',
            #'likly_cenR', 'likly_nccR', 'likly_sobR', 'likly_sadR',
            ]
    d0 = 10
    imgs_to_plot = []
    tmp_fn = lambda idx, img : np.argmin(img, axis = 0) if idx < 4 else np.argmax(img, axis = 0)
    #tmp_fn = lambda idx, img : img[d0,:,:] if idx < 4 else np.argmax(img, axis = 0)
    for k in range(8):
        #imgs_to_plot.append(cbmv_features[k,d0, :,:])
        imgs_to_plot.append(tmp_fn(k, cbmv_features[k,:,:,:]))
    
    show_4_imgs_2_row(imgs = imgs_to_plot, 
                img_names = ['slice_{}_d{}'.format(i, d0) for i in cbmv_features_name],
                cmap = ['inferno']*8
                #cmap = ['gray']*8
                )
    for k in range(0, len(imgs_to_plot)):
        print ('feature {} : {:>10}, min = {}, max = {}'.format(k, cbmv_features_name[k], 
                np.amin(imgs_to_plot[k]), np.amax(imgs_to_plot[k])))

""" Generate start_w, start_h for cropping image for network training:
    h: image height;
    w: image width;
    crop_height: crop height;
    crop_width: crop width;
    board_w_left: left width board;
    board_w_right: right width boar ;
    board_h: hight board;
"""
def get_crop_position(w, h, crop_width = 512, 
    crop_height = 256, board_w_left = 256, 
    board_w_right= 0, board_h = 10, 
    #For validation cases, we sometimes would like to fix the 
    # cropping position and try to crop the image around its center;
    is_fixed_center_around_crop = False
    ):

    #NOTE: updated on Sep 4, 2019:
    # updated for eth3d and kt12 training together,
    # duet to eth image has small width than kt12
    tmp_diff = w-crop_width - board_w_left -board_w_right
    #print ("[???] 1 tmp_diff = ", tmp_diff)
    if tmp_diff >= 0:
        new_board_w_left = board_w_left
        new_board_w_right = board_w_right
    else:
        while tmp_diff < 0:
            new_board_w_left = board_w_left // 2
            new_board_w_right = board_w_right // 2 # newly added for right->left direction!
            tmp_diff = w-crop_width - new_board_w_left - new_board_w_right
            #print ("[???] tmp_diff = ", tmp_diff)

    start_w = random.randint(0, w-crop_width- new_board_w_left - new_board_w_right)
    start_h = random.randint(0, h-crop_height-2*board_h)

    """ When we want to keep the cropping position and try to crop the image around its center;"""
    if is_fixed_center_around_crop:
        start_w = (w - crop_width  - new_board_w_left - new_board_w_right)//2 - 1
        start_h = (h - crop_height - 2*board_h)//2 - 1
    
    finish_h = start_h + (crop_height + 2*board_h) # '2*10' is used as border
    finish_w = start_w + (crop_width + new_board_w_left + new_board_w_right)
    #print ("[???] Cropped at start_w=%d, start_h=%d, finish_w=%d, finish_h=%d" %(start_w, start_h, finish_w, finish_h))
    return start_w, start_h, finish_w, finish_h, new_board_w_left, new_board_w_right

def get_default_args_dict():
    args_dict = {
        # from hp
        "censw": 11,
	"nccw": 3,
	"sadw": 5,
	"sobelw": 5,
        "cens_sigma": 128.0,
	"ncc_sigma": 0.02,
        "sad_sigma": 20000.0,
        "sobel_sigma": 20000.0,
        "cbmv_F": 8,
        #"cbmv_F": 4,
        
        #from up
	"w_padding":   1248,
        "h_padding":  384,
	"board_h": 12,
        "seed": 1234,
        "batch_h": 256,
        "overlap_board": 20,
        #"overlap_board": 30,
        "batch_in_image": 0,
        #'ds_scale': 4,
        'ds_scale': 2,
        'sf_frames_type': 'frames_finalpass',
        #'sf_frames_type': 'frames_cleanpass',
    }
    return args_dict


def down_sampling_input(ds_scale, imgl, imgr, anti_aliasing=True, 
                        multichannel = False, preserve_range = True):
    #max_l = np.amax(imgl)
    #max_r = np.amax(imgr)
    max_l = 255.0
    imgl = skimage.transform.rescale(image = imgl.astype(np.float32)/max_l, 
                scale = ds_scale, anti_aliasing=anti_aliasing, 
                preserve_range = preserve_range, multichannel = multichannel, 
                mode = 'constant')
    imgl = (imgl*max_l).astype(np.uint8).copy(order='C')
    
    max_r = 255.0
    imgr = skimage.transform.rescale(image = imgr.astype(np.float32)/max_r, 
                scale = ds_scale, anti_aliasing=anti_aliasing, 
                preserve_range = preserve_range, multichannel = multichannel, 
                mode = 'constant')
    imgr = (imgr*max_r).astype(np.uint8).copy(order='C')
    return imgl, imgr


def remove_border(
    src_arr, # input array
    h_s, # height idx start
    h_e, # height idx end
    w_s, # width idx start
    w_e # width idx end
    ):
    assert(src_arr.ndim == 2 or src_arr.ndim == 3)
    if h_e is not None:
        res = src_arr[h_s:h_e, :]
    else:
        res = src_arr[h_s:, :]
    
    if w_e is not None:
        res = res[:, w_s: w_e]
    else:
        res = res[:, w_s: ]
    # change to contiguous array
    res = np.ascontiguousarray(res)
    return res

""" dummy CBMV cost volume data-generator """
# for training, we should do image cropping randomly;
def generate_dummy_crop_train_cbmv(
    limg_name, # left image name
    rimg_name, # right image name
    disp_name, # left disparity file name
    lseg_name, # left segmentation gt file name
    crop_height = 256,
    crop_width = 512,
    maxdisp = 192,
    is_fixed_center_around_crop = False,
    args_dict = None,
    #NOTE: This parameter is added for iResNet case, because of the left-right feature constancy used in disparity refinement!!!
    is_left_only = True, # only generate feature for left image; otherwise, for both left and right;
    ):
    #print ("[?????????????????] generate_dummy_crop_train_cbmv ... Just for code debugging !!!!") 
    if args_dict is None:
        args_dict = get_default_args_dict()
    
    ds_scale = 1.0/ float(args_dict['ds_scale']) # e.g., == 4; down-smapling scale for input images;
    tmpH = crop_height // args_dict['ds_scale']
    tmpW = crop_width // args_dict['ds_scale']
    tmpD = maxdisp // args_dict['ds_scale']

    #print ('[***] ds_scale = {}'.format(ds_scale))
    disp_image = 64.0*np.ones((crop_height, crop_width), order='C')
    imgl_rgb = np.zeros((3,crop_height, crop_width), order='C')

    """ features shape : [channel (e.g., = 8, or 16), ndisp, imgH, imgW] """
    if is_left_only:
        features = np.zeros((8, tmpD, tmpH, tmpW), order= 'C')
    else:
        features = np.zeros((16, tmpD, tmpH, tmpW), order= 'C')

    features = torch.from_numpy(features).float()
    disp_image = torch.from_numpy(disp_image).float()
    imgl_rgb = torch.from_numpy(imgl_rgb).float()
    semantic_label = torch.zeros([1, disp_image.size()[0], disp_image.size(1)], dtype=torch.float32)
    
    return features, disp_image, imgl_rgb, semantic_label

""" CBMV cost volume data-generator """
# for training, we should do image cropping randomly;
def generate_crop_train_cbmv(
    limg_name, # left image name
    rimg_name, # right image name
    disp_name, # left disparity file name
    lseg_name, # left segmentation gt file name
    crop_height = 256,
    crop_width = 512,
    maxdisp = 192,
    is_fixed_center_around_crop = False,
    args_dict = None,
    #NOTE: This parameter is added for iResNet case, because of the left-right feature constancy used in disparity refinement!!!
    is_left_only = True # only generate feature for left image; otherwise, for both left and right;
    ):
    
    if args_dict is None:
        args_dict = get_default_args_dict()
    
    censw = args_dict['censw'] # e.g., == 11; 
    nccw = args_dict['nccw'] # e.g., == 3; 
    sadw = args_dict['sadw'] # e.g., == 5; 
    sobelw = args_dict['sobelw'] # e.g., == 5; 

    cens_sigma = args_dict['cens_sigma']
    ncc_sigma = args_dict['ncc_sigma']
    sad_sigma = args_dict['sad_sigma']
    sobel_sigma = args_dict['sobel_sigma']
    board_h = args_dict['board_h']

    #NOTE: board_w is related to unmatchable region, 
    # in theory board_w >= max_disp to avoid unmatchable region around 
    # image boundary (left boundary for left iamge, and 
    # right boundary for right one).
    board_w = maxdisp

    """ Updated on 01/19/2019: 
        if only left->right direction, we set board_w_left = maxdisp, 
        e.g., = 192, and board_w_right = 0 or a small value """
    board_w_left = board_w
    """ Updated on 02/11/2020: 
        if also considering right->left direction, we aslo set board_w_right = maxdisp; 
         or a small value if image width is not large enough """
    if is_left_only:
        board_w_right = 0
    else:
        board_w_right = board_w

    
    ds_scale = 1.0/ float(args_dict['ds_scale']) # e.g., == 4; down-smapling scale for input images;
    #print ('[***] ds_scale = {}'.format(ds_scale))
    
    """ disable this seed setting, due to the seed which has been set in main python module """
    #if args_dict['seed'] > 0: 
    #    random.seed(args_dict['seed'])
    #else:
    #    random.seed(datetime.now())
    
    """ as np.uint8, because the matcher wrapper requires it"""
    #print ('[**] {},{},{}'.format(limg_name, rimg_name, disp_name))
    #NOTE: For CBMV featrue extraction, the input image should be guaranteed 
    # in type UINT8 and in gray image (i.e., 1 channel);
    # Load an color image in grayscale
    imgl = cv2.imread(limg_name, 0).astype(np.uint8)
    imgl_rgb = cv2.imread(limg_name, 1).astype(np.uint8)[:,:,::-1]
    imgr = cv2.imread(rimg_name, 0).astype(np.uint8)
    imgr_rgb = cv2.imread(rimg_name, 1).astype(np.uint8)[:,:,::-1]
    h, w = imgl.shape[:2] 
    #print ('[???] read {}, imgl shape = {}'.format(limg_name, imgl.shape))
    #print ('[???] imgl_rgb shape = ', imgl_rgb.shape)
    # not starting at (0, 0), to avoid RAND_MAX values on border 
    start_w, start_h, finish_w, finish_h, board_w_left, board_w_right = get_crop_position(
        w, h, crop_width, crop_height, board_w_left, 
        board_w_right, board_h, is_fixed_center_around_crop)
    
    disp_image = pfm.readPFM(disp_name)
    disp_image = disp_image[start_h:finish_h, start_w:finish_w]
    disp_image[disp_image == np.inf] = .0
    
    #pfm.show(imgl, title='imgl')
    #pfm.show(imgr, title='imgr')
    #pfm.show_uint8(imgl_rgb, title='imgl_rgb')
    #pfm.show(disp_image, title='disp GT')

    """ remove the 20-pixel border """
    #valid ending index for width;
    vld_w_end = - board_w_right if board_w_right > 0 else None
    # valid ending index for height;
    vld_h_end = - board_h if board_h > 0 else None

    
    disp_image = remove_border(disp_image, board_h, vld_h_end, board_w_left, vld_w_end)
    #print ("[???] after remove boarder disp_image : ", disp_image.shape)
    # add channel dim to change [H,W] to [C=1,H, W]
    #disp_image = disp_image[None,:,:]
    
    #imgl_rgb = imgl_rgb[board_h: vld_h_end, board_w_left: vld_w_end, :]
    imgl_rgb = imgl_rgb[start_h:finish_h, start_w:finish_w, :]
    imgl_rgb = remove_border(imgl_rgb, board_h, vld_h_end, board_w_left, vld_w_end)
    imgr_rgb = imgr_rgb[start_h:finish_h, start_w:finish_w, :]
    imgr_rgb = remove_border(imgr_rgb, board_h, vld_h_end, board_w_left, vld_w_end)
    #print ("[???] after remove boarder imgl_rgb : ", imgl_rgb.shape)

    if lseg_name is not None:
        semantic_label = np.asarray(Image.open(lseg_name), dtype=np.float32, order="C")
        semantic_label = semantic_label[start_h:finish_h, start_w:finish_w]
        semantic_label = remove_border(semantic_label, board_h, vld_h_end, board_w_left, vld_w_end)
    else:
        semantic_label = None

    
    #NOTE: make sure the cropped images has contiguous memory.
    #      Since the following ncc, census etc matacher requires that.
    imgl = imgl[start_h:finish_h, start_w:finish_w].copy(order='C')
    imgr = imgr[start_h:finish_h, start_w:finish_w].copy(order='C')
    
    if ds_scale < 1.0:
        imgl, imgr = down_sampling_input(ds_scale, imgl, imgr)
        
    #print ("[???] maxdisp=%d,censw=%d,nccw=%d,sadw=%d,sobelw=%d"%(maxdisp, censw, nccw, sadw, sobelw))
    census, ncc, sobel, sad = get_costs(
               imgl,
               imgr, 
               maxdisp// args_dict['ds_scale'], 
               censw, nccw, sadw, sobelw, 
               board_h// args_dict['ds_scale'], 
               board_w_left// args_dict['ds_scale'], 
               board_w_right// args_dict['ds_scale']
               )
    
    #print ("[????] census shape = ", census.shape)
    
    """ features shape : [channel (e.g., = 8, or 16), ndisp, imgH, imgW] """
    if is_left_only:
        features = extract_features_left(census, ncc, sobel, sad, cens_sigma, 
                                     ncc_sigma, sad_sigma, sobel_sigma)
    else:
        features = extract_features_lr(census, ncc, sobel, sad, cens_sigma, 
                                     ncc_sigma, sad_sigma, sobel_sigma)


    """ change it from [C=8, imgH, imgW, ndisp] to [C=8, disp, imgH, imgW] """
    #show_2_imgs_1_row( imgl, imgr)
    #pfm.show_uint8(imgl_rgb, title='imgl_rgb')
    #print ('imgl shape = ', imgl.shape, 'imgr shape = ', imgr.shape,)
    # change channel first
    imgl_rgb = imgl_rgb.transpose((2,0,1)).astype(np.float32) / 255.0
    imgr_rgb = imgr_rgb.transpose((2,0,1)).astype(np.float32) / 255.0
    
    """for debugging, visualize the training data """
    #debug_cbmv_featues(features)
    #print ("[???] features shape = ", features.shape)
    
    features = torch.from_numpy(features).float()
    disp_image = torch.from_numpy(disp_image).float()
    #print ("[???] to tensor imgl_rgb : ", imgl_rgb.shape)
    imgl_rgb = torch.from_numpy(imgl_rgb).float()
    imgr_rgb = torch.from_numpy(imgr_rgb).float()

    del census
    del ncc
    del sobel
    del sad 
    # > see: https://discuss.pytorch.org/t/getting-the-some-of-the-strides-of-a-numpy-error-are-negative-on-using-facebook-infersent-model/31837;
    #NOTE: do not forget to use torch.from_numpy() to convert numpy arrays to Tensor 
    # before giving them to pytorch’s function, due to the  the fact that not all numpy 
    # arrays can be represented as Tensor (arrays that were flipped in particular). 
    # You can use np.ascontiguousarray() before giving your array to pytorch 
    # to make sure it will work.
    if semantic_label is not None:
       # add channel dim to change [H,W] to [C=1,H, W]
       semantic_label = troch.from_numpy(semantic_label[None, ...]).float()
    else:
       #just use np.zeros()
       semantic_label = torch.zeros([1, disp_image.size()[0], disp_image.size(1)], 
                                    dtype=torch.float32)
    return features, disp_image, imgl_rgb, imgr_rgb, semantic_label


# for test, we generate cbmv features per image;
def generate_test_cbmv(
    limg_name, # left image name
    rimg_name, # right image name
    #disp_name, # left disparity file name
    #lseg_name, # left segmentation gt file name
    crop_height = 384,
    crop_width = 1248,
    encoder_ds = 64, # e.g., encoder downscale up to 1/64 for dispnetc and iresnet
    maxdisp = 192,
    args_dict = None,
    #NOTE: This parameter is added for iResNet case, because of the left-right feature constancy used in disparity refinement!!!
    is_left_only = True, # only generate feature for left image; otherwise, for both left and right;
    ):
    
    if args_dict is None:
        args_dict = get_default_args_dict()
    
    censw = args_dict['censw'] # e.g., == 11; 
    nccw = args_dict['nccw'] # e.g., == 3; 
    sadw = args_dict['sadw'] # e.g., == 5; 
    sobelw = args_dict['sobelw'] # e.g., == 5; 

    cens_sigma = args_dict['cens_sigma']
    ncc_sigma = args_dict['ncc_sigma']
    sad_sigma = args_dict['sad_sigma']
    sobel_sigma = args_dict['sobel_sigma']
    ds_scale = 1.0/ float(args_dict['ds_scale']) # e.g., == 4; down-smapling scale for input images;
    #print ('[***] ds_scale = {}'.format(ds_scale))
    
    """ as np.uint8, because the matcher wrapper requires it"""
    #print ('[**] {},{},{}'.format(limg_name, rimg_name, disp_name))
    #NOTE: For CBMV featrue extraction, the input image should be guaranteed 
    # in type UINT8 and in gray image (i.e., 1 channel);
    # Load an color image in grayscale
    print ("[??] reading ", limg_name) # left image name
    imgl = cv2.imread(limg_name, 0).astype(np.uint8)
    imgr = cv2.imread(rimg_name, 0).astype(np.uint8)
    h, w = imgl.shape[:2]
    
    #if dispname != '' and os.path.isfile(dispname):
    #    dispGT = pfm.readPFM(dispname)
    #    dispGT[dispGT == np.inf] = .0
    #else:
    #    dispGT= None
    
    #if lseg_name is not None:
    #    semantic_label = np.asarray(Image.open(lseg_name), dtype=np.float32, order="C")
    #else:
    #    semantic_label = None

    # do padding on left side and top side;
    # due to eth3d images have diverse image size;
    # so we process each of them seperately;
    crop_width =  w + (encoder_ds - w % encoder_ds) % encoder_ds
    crop_height = h + (encoder_ds - h % encoder_ds) % encoder_ds
    assert h <= crop_height and w <= crop_width
    #print ("[???]input HxW=%dx%d, padding size=%dx%d" %(h,w,crop_height, crop_width))
    pad_w = crop_width - w
    pad_h = crop_height - h
    """ #Updated@2020/03/11: padding to upper-right direction """
    imgl = np.pad(imgl, ((pad_h, 0), (0, pad_w)), 'constant').astype(np.uint8).copy(order='C')
    imgr = np.pad(imgr, ((pad_h, 0), (0, pad_w)), 'constant').astype(np.uint8).copy(order='C')

    #elif h > crop_height and w > crop_width:
    #    start_x = max((w - crop_width) // 2, 0)
    #    start_y = max((h - crop_height)// 2, 0)
    #    imgl = imgl[start_y: start_y + crop_height, start_x: start_x + crop_width].astype(np.uint8).copy(order='C')
    #    imgr = imgr[start_y: start_y + crop_height, start_x: start_x + crop_width].astype(np.uint8).copy(order='C')
    #else:
    #    raise ValueError("crop size is not correct!!")
    
    #pfm.show(imgl, title='imgl')
    #pfm.show(imgr, title='imgr')
    #pfm.show(disp_image, title='disp GT')

    if ds_scale < 1.0:
        imgl, imgr = down_sampling_input(ds_scale, imgl, imgr)
    #print ("[???] maxdisp=%d,censw=%d,nccw=%d,sadw=%d,sobelw=%d"%(maxdisp, censw, nccw, sadw, sobelw))
    """ NOTE:??? for testing cbmv, we do not do boarding padding """
    #if 0:
    #    board_h = 0
    #    board_w_left = 0
    #    board_w_right = 0
    #    census, ncc, sobel, sad = get_costs(imgl,imgr, maxdisp// args_dict['ds_scale'], 
    #            censw, nccw, sadw, sobelw, 
    #            board_h, 
    #            board_w_left, 
    #            board_w_right
    #            )
    
    #UPDATED: @2020-03-11:
    # padding board_w and board_h, to avoid RAND_MAX values on border
    board_h = 10 # for generating test cbmv, we just set the board_h = 10;
    #board_w = maxdisp/args_dict['ds_scale']
    board_w = 10
    imgl_board = np.pad(imgl, ((board_h, board_h), (board_w, board_w)), 'constant').astype(np.uint8).copy(order='C')
    imgr_board = np.pad(imgr, ((board_h, board_h), (board_w, board_w)), 'constant').astype(np.uint8).copy(order='C')
    #print ('imgl shape = {} by {}'.format(imgl.shape[0], imgl.shape[1]))
    #print ('imgl_board shape = {} by {}'.format(imgl_board.shape[0], imgl_board.shape[1]))
    census, ncc, sobel, sad = get_costs(
               imgl_board,
               imgr_board, 
               maxdisp// args_dict['ds_scale'], 
               censw, nccw, sadw, sobelw, 
               board_h, 
               board_w, #board_w_left
               board_w #board_w_right
               )
    
    #print ("[????] census shape = ", census.shape)
    """ features shape : [channel (e.g., = 8, or 16), ndisp, imgH, imgW] """
    if is_left_only:
        features = extract_features_left(census, ncc, sobel, sad, cens_sigma, 
                                     ncc_sigma, sad_sigma, sobel_sigma)
    else:
        features = extract_features_lr(census, ncc, sobel, sad, cens_sigma, 
                                     ncc_sigma, sad_sigma, sobel_sigma)

    """for debugging, visualize the testing data """
    #debug_cbmv_featues(features)
    #print ("[???] features shape = ", features.shape)
    #sys.exit()
    
    del census
    del ncc
    del sobel
    del sad
    
    """ change it from [C=8, imgH, imgW, ndisp] to [C=8, disp, imgH, imgW] """
    #show_2_imgs_1_row( imgl, imgr)
    #pfm.show_uint8(imgl_rgb, title='imgl_rgb')
    #print ('imgl shape = ', imgl.shape, 'imgr shape = ', imgr.shape,)
    
    features = torch.from_numpy(features).float()
    return features, h, w, crop_height, crop_width
