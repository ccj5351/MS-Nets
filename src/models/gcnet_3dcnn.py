# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: gcnet_3dcnn.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 12-12-2019
# @last modified: Thu 13 Feb 2020 12:19:40 AM EST

# > see: the code is adapoted from https://github.com/zyf12389/GC-Net/blob/master/gc_net.py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .net_init import net_init


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))

def deconvbn_3d(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, 
                         padding=1, output_padding=1, stride=2, bias=False),
                         nn.BatchNorm3d(out_planes))


class Conv3DBlock(nn.Module):
    def __init__(self,in_planes,planes,stride=1,kernel_size =3):
        super(Conv3DBlock, self).__init__()
        #self.conv1 = nn.Conv3d(in_planes, planes, kernel_size, stride=stride, padding=1)
        #self.bn1 = nn.BatchNorm3d(planes)
        self.convbn_3d_1 = convbn_3d(in_planes, planes, kernel_size, stride=stride, pad = 1)
        self.convbn_3d_2 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.convbn_3d_3 = convbn_3d(planes, planes, kernel_size, stride=1, pad = 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.convbn_3d_1(x))
        out = self.relu(self.convbn_3d_2(out))
        out = self.relu(self.convbn_3d_3(out))
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)
    
    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out


class GCNet_CostVolumeAggre(nn.Module):
    def __init__(self, 
            maxdisp = 192, 
            cbmv_in_planes = 8, 
            kernel_size = 3, 
            # if True, means the input cost volume is in size [8, D/4, H/4, W/4];
            is_quarter_input_size = False,
            #is_quarter_input_size = True,
            ):
        super(GCNet_CostVolumeAggre, self).__init__()
        self.maxdisp = maxdisp
        self.F = 32 # e.g., == 32;
        self.kernel_size = kernel_size # e.g., == 3;
        self.relu = nn.ReLU(inplace=True)
        #conv3d
        self.conv3dbn_1 = convbn_3d(cbmv_in_planes, self.F, self.kernel_size, stride= 1, pad = 1)
        self.conv3dbn_2 = convbn_3d(self.F, self.F, self.kernel_size, stride= 1, pad = 1)

        #conv3d sub_sample block
        self.block_3d_1 = Conv3DBlock(self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
        self.block_3d_2 = Conv3DBlock(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
        self.block_3d_3 = Conv3DBlock(2*self.F, 2*self.F, stride=2, kernel_size=self.kernel_size)
        self.block_3d_4 = Conv3DBlock(2*self.F, 4*self.F, stride=2, kernel_size=self.kernel_size)
        
        #deconv3d, with BN and ReLU
        self.deconvbn1 = deconvbn_3d(4*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn2 = deconvbn_3d(2*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn3 = deconvbn_3d(2*self.F, 2*self.F, self.kernel_size, stride=2)
        self.deconvbn4 = deconvbn_3d(2*self.F, self.F, self.kernel_size, stride=2)

        #last deconv3d, no BN or ReLU
        if is_quarter_input_size:
            print ("[***] Input Cost volume is in quarter size !!!")
            self.deconv5 = nn.ConvTranspose3d(self.F, 1, kernel_size=3, stride=4, padding=1, output_padding=3)
        else:
            self.deconv5 = nn.ConvTranspose3d(self.F, 1, self.kernel_size, stride=2, padding=1, output_padding=1)
        
        net_init(self)
        print ("[***]GCNet_CostVolumeAggre() weights inilization done!")

    def forward(self, cv):
        """
        args:
            cost: cost volume in size [N,C,D,H,W]
        """
        #print ("[???] cv shape: ", cv.shape)
        out = self.relu(self.conv3dbn_1(cv)) # conv3d_19
        out = self.relu(self.conv3dbn_2(out)) # conv3d_20
        
        #conv3d block
        res_l20 = out # from layer conv3d_20;
        out = self.block_3d_1(out) # conv3d_21,22,23
        res_l23 = out
        out = self.block_3d_2(out) # conv3d_24,25,26
        res_l26 = out
        out = self.block_3d_3(out) # conv3d_27,28,29
        res_l29 = out
        out = self.block_3d_4(out) # conv3d_30,31,32
        #print ("[???] after conv3d_32 out shape = ", out.shape)
        
        #deconv3d
        #print ("[???] res_l29: ", res_l29.shape)
        out = self.relu(self.deconvbn1(out) + res_l29)
        out = self.relu(self.deconvbn2(out) + res_l26)
        out = self.relu(self.deconvbn3(out) + res_l23)
        out = self.relu(self.deconvbn4(out) + res_l20)
        #last deconv3d, no BN or ReLU
        out = self.deconv5(out) # [N, 1, D, H, W]
        #print ("[???] out shape = ", out.shape)
        out = torch.squeeze(out, dim = 1) 
        prob = F.softmax(out,1)
        disp = self.disparityregression(prob)
        #disp = disparityregression(self.maxdisp)(prob)
        return disp

    def disparityregression(self, x):
        #with torch.cuda.device_of(x):
        N, D, H, W = x.size()[:]
        assert D == self.maxdisp, "%d != %d"% (D, self.maxdisp)
        disp = torch.tensor(np.array(range(self.maxdisp)), dtype=torch.float32, 
                requires_grad=False).cuda().view(1,self.maxdisp,1,1)
        disp = disp.repeat(N,1,H,W)
        disp = torch.sum(x*disp, 1) # in size [N, H, W] as output
        #print ("[???] final disp  : ", disp.size())
        return disp
