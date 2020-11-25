# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file:
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 22-02-2018
# @last modified: Fri 25 Jan 2019 02:16:47 PM EST

import sys
#sys.path.insert(0,'.')
import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_4_imgs_4_row( imgs, img_names, cmap = ['gray']*16):
    fig = plt.figure()
    for i in range(0, len(imgs)):
        a = fig.add_subplot(4,4,i+1)
        print ('i = ', i)
        imgplot = plt.imshow(imgs[i].astype(np.float32), cmap = cmap[i])
        #imgplot.set_clim(0.0,0.7)
        a.set_title(img_names[i])
        #plt.colorbar(ticks=[0.1, 0.3,0.5,0.7], orientation='horizontal')
    plt.show()

def show_4_imgs_3_row( imgs, img_names, cmap = ['gray']*12):
    fig = plt.figure()
    for i in range(0, len(imgs)):
        a = fig.add_subplot(3,4,i+1)
        imgplot = plt.imshow(imgs[i].astype(np.float32), cmap = cmap[i])
        #imgplot.set_clim(0.0,0.7)
        a.set_title(img_names[i])
        #plt.colorbar(ticks=[0.1, 0.3,0.5,0.7], orientation='horizontal')
    plt.show()

def show_4_imgs_2_row( imgs, img_names, cmap = ['gray']*8):
    fig = plt.figure()
    for i in range(0, len(imgs)):
        a = fig.add_subplot(2,4,i+1)
        #imgplot = plt.imshow(imgs[i].astype(np.float32), cmap = cmap[i])
        imgplot = plt.imshow(imgs[i], cmap = cmap[i])
        a.set_title(img_names[i])
    plt.show()

def show_imgs(imgs, img_names, rows, cols, BGR2RGB = None, cmap = None ):
    fig = plt.figure()
    assert len(imgs) == rows*cols
    for i in range(0, rows*cols):
        a = fig.add_subplot(rows,cols,i+1)
        #imgplot = plt.imshow(imgs[i].astype(np.float32), cmap = cmap[i])
        tmp_img = imgs[i]
        if BGR2RGB and len(tmp_img.shape) == 3:
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        imgplot = plt.imshow(tmp_img, cmap = cmap[i])
        a.set_title(img_names[i])
    plt.show()

def show_3_imgs_1_row(
        left_img, # 2d array
        mid_img, # 2d array
        right_img, # 2d array
        left_img_name = 'left iamge', 
        mid_img_name = 'middle iamge', 
        right_img_name = 'right iamge', 
        cmap = ['gray', 'gray', 'gray']
        ):
    fig = plt.figure()
    print ('left shape = ', left_img.shape, 'mid_img shape = ', mid_img.shape, 'right shape = ', right_img.shape)
    a = fig.add_subplot(1,3,1)
    imgplot = plt.imshow(left_img.astype(np.float32), cmap = cmap[0])
    #imgplot.set_clim(0.0,0.7)
    a.set_title(left_img_name)
    #plt.colorbar(ticks=[0.1, 0.3,0.5,0.7], orientation='horizontal')

    a = fig.add_subplot(1,3,2)
    imgplot = plt.imshow(mid_img.astype(np.float32), cmap = cmap[1])
    #imgplot.set_clim(0.0,0.7)
    a.set_title(mid_img_name)
    #plt.colorbar(ticks=[0.1, 0.3,0.5,0.7], orientation='horizontal')


    a = fig.add_subplot(1,3,3)
    imgplot = plt.imshow(right_img.astype(np.float32), cmap = cmap[2])
    #imgplot.set_clim(0.0,0.7)
    a.set_title(right_img_name)
    #plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='horizontal')
    plt.show()

def show_2_imgs_1_row(
        left_img, # 2d array
        right_img, # 2d array
        left_img_name = 'left iamge', 
        right_img_name = 'right iamge', 
        cmap = 'gray'
        ):
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(left_img.astype(np.float32), cmap = cmap)
    #imgplot.set_clim(0.0,0.7)
    a.set_title(left_img_name)
    #plt.colorbar(ticks=[0.1, 0.3,0.5,0.7], orientation='horizontal')

    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(right_img.astype(np.float32), cmap = cmap)
    #imgplot.set_clim(0.0,0.7)
    a.set_title(right_img_name)
    #plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='horizontal')
    plt.show()


#-------------------------
# some utility functions
#-------------------------
def print_ms_gcnet_params(model):
    print('Number of MS-GCNet model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('Including:\n1) number of Feature Extraction module parameters: {}'.format(
        sum(
            [p.data.nelement() for n, p in model.named_parameters() if any(
                ['module.convbn0' in n, 
                 'module.res_block' in n, 
                 'module.conv1' in n
                 ])] )))

    print('2) number of Other modules parameters: {}'.format(
        sum(
            [p.data.nelement() for n, p in model.named_parameters() if any(
                ['module.conv3dbn' in n,
                 'module.block_3d' in n,
                 'module.deconv' in n,
                 ])] )))
    
    for i, (n, p) in enumerate(model.named_parameters()):
        print (i, "  layer ", n, "has # param : ", p.data.nelement())


def get_dataloader_len(train_file_list_txt, batch_size):
    f = open(train_file_list_txt, 'r')
    file_list = [l.rstrip() for l in f.readlines()]
    n = len(file_list)
    n_per_batch = n // batch_size
    print ("[***]Iterable Dataset!!! #Img=%d,batch_size=%d,data_loader_len=%d"%(
        n, batch_size, n_per_batch))
    return  n_per_batch