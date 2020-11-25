# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: dispnet_2dcnn.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 18-02-2020
# @last modified: Fri 28 Feb 2020 01:31:47 PM EST

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#Convolution In/Out Size:
#O = floor{(W - F + 2P)/S + 1}

#def correlation1D_map(x, y, maxdisp=40):
class correlation1D_map_V1(nn.Module):
    def __init__(self, maxdisp):
        super(correlation1D_map_V1, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x, y):
        """
        args:
            x: left feature, in [N,C,H,W]
            y: right feature, in [N,C,H,W]
            max_disp: disparity range
        return:
            corr: correlation map in size [N,D,H,W]
        """
    
        """
        #NOTE: make sure x means left image, y means right image,
        # so that we have the pixel x in left image, 
        # and the corresponding match pixel in right image has x-d 
        # (that means disparity d = shifting to left by d pixels). 
        # Otherwise, we have to chagne the shift direction!!!
        """
        # Pads the input tensor boundaries with zero.
        # padding = (padding_left, padding_right, padding_top, padding_bottom) along the [H, W] dim; 
        #y_pad = nn.ZeroPad2d((self.maxdisp-1, 0, 0, 0))(y)
        #NOTE: updated maxdisp to maxdisp-1 for left padding!!!
        y_pad = nn.ZeroPad2d((self.maxdisp-1, 0, 0, 0))(y)
        # input width
        W0 = x.size()[3]
        corr_tensor_list = []
        #NOTE: reversed() is necessary!!!
        for d in reversed(range(self.maxdisp)):
            x_slice = x
            #added by CCJ:
            #Note that you don’t need to use torch.narrow or select, 
            #but instead basic indexing will do it for you.
            y_slice = y_pad[:,:,:,d:d+W0]
            #xy_cor = torch.mean(x_slice*y_slice, dim=1, keepdim=True)
            #NOTE: change the mean to sum!!!
            xy_cor = torch.sum(x_slice*y_slice, dim=1, keepdim=True)
            #CosineSimilarity
            #cos = nn.CosineSimilarity(dim=1, eps=1e-08)
            #xy_cor = torch.unsqueeze(cos(x_slice,y_slice),1)
            corr_tensor_list.append(xy_cor)
        corr = torch.cat(corr_tensor_list, dim = 1)
        #print ("[???] corr shape: ", corr.shape)
        return corr

# corr1d
# code is adapted from https://github.com/wyf2017/DSMnet/blob/master/models/util_conv.py
class Corr1d_V2(nn.Module):
    def __init__(self, kernel_size=1, stride=1, D=1, simfun=None):
        super(Corr1d_V2, self).__init__()
        
        self.kernel_size = kernel_size
        if kernel_size > 1:
            assert kernel_size%2 == 1
            self.avg_func = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.stride = stride
        self.D = D
        if(simfun is None):
            self.simfun = self.simfun_default
        else: # such as simfun = nn.CosineSimilarity(dim=1)
            self.simfun = simfun
    
    def simfun_default(self, fL, fR):
        return (fL*fR).sum(dim=1)
        
    def forward(self, fL, fR):
        bn, c, h, w = fL.shape
        D = self.D
        stride = self.stride
        kernel_size = self.kernel_size
        corrmap = Variable(torch.zeros(bn, D, h, w).type_as(fL.data))
        corrmap[:, 0] = self.simfun(fL, fR)
        for i in range(1, D):
            if(i >= w): break
            idx = i*stride
            corrmap[:, i, :, idx:] = self.simfun(fL[:, :, :, idx:], fR[:, :, :, :-idx])
        if(kernel_size>1):
            corrmap = self.avg_func(corrmap)
        return corrmap



""" NO BatchNorm Version """
def downsample_conv(in_planes, out_planes, kernel_size = 3):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding = (kernel_size-1)//2),
            nn.ReLU(inplace=True)
           )

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True), #bias=True by default;
            nn.ReLU(inplace=True)
           )

def conv1x1(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
           )

#O = (W − 1)×S − 2P + K + output_padding
def upconv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
           )

def upconv4x4(in_planes, out_planes):
    return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True)
           )

#2d convolution with padding, bn and activefun
def downsample_conv_bn(in_planes, out_planes, kernel_size = 3, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    assert kernel_size % 2 == 1
    myconv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding = (kernel_size-1)//2, bias = bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] downsample_conv_bn() Enable BN")
    else:
        print ("[**] downsample_conv_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
    else:
        print ("[**] downsample_conv_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


def conv3x3_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias) 
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] conv3x3_bn() Enable BN")
    else:
        print ("[**] conv3x3_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] conv3x3_bn() Enable ReLU")
    else:
        print ("[**] conv3x3_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)

def conv1x1_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias= bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] conv1x1_bn() Enable BN")
    else:
        print ("[**] conv1x1_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] conv1x1_bn() Enable ReLU")
    else:
        print ("[**] conv1x1_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


def upconv3x3_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1, bias= bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] upconv3x3_bn() Enable BN")
    else:
        print ("[**] upconv3x3_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] upconv3x3_bn() Enable ReLU")
    else:
        print ("[**] upconv3x3_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


def upconv4x4_bn(in_planes, out_planes, is_relu = True, is_bn = True):
    if is_bn:
        bias = False
    else:
        bias = True
    myconv2d = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
    if (not is_bn and not is_relu):
        return myconv2d
    layers = []
    layers.append(myconv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
        #print ("[**] upconv4x4_bn() Enable BN")
    else:
        print ("[**] upconv4x4_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
        #print ("[**] upconv4x4_bn() Enable ReLU")
    else:
        print ("[**] upconv4x4_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)

def deconv2d_bn(in_planes, out_planes, kernel_size=4, stride=2, is_bn = True, is_relu = True):
    "2d deconvolution with padding, bn and relu"
    assert stride > 1
    p = (kernel_size - 1) // 2
    op = stride - (kernel_size - 2*p)
    if is_bn:
        bias = False
    else:
        bias = True
    
    conv2d = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding = p,  output_padding = op, bias = bias)
    
    if(not is_bn and not is_relu): 
        return conv2d
    
    layers = []
    layers.append(conv2d)
    if is_bn:
        layers.append(nn.BatchNorm2d(out_planes))
    else:
        print ("[**] deconv2d_bn() DISABLE BN !!!")
    if is_relu:
        layers.append(nn.ReLU(inplace=True))
    else:
        print ("[**] deconv2d_bn() DISABLE ReLU !!!")
    return nn.Sequential(*layers)


class disparityregression(nn.Module):
    def __init__(self, maxdisp, sumKeepDim=False):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)
        self.sumKeepDim = sumKeepDim
    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1, keepdim=self.sumKeepDim)
        return out

def deconvbn_3d(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, 
                    padding=1, output_padding=1, stride=2, bias=False), 
                    nn.BatchNorm3d(out_planes)) 