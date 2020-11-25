# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: psmnet_3dcnn.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 12-12-2019
# @last modified: Mon 13 Jan 2020 02:01:05 AM EST

# > see: the code is adapoted from https://github.com/zyf12389/GC-Net/blob/master/gc_net.py
from __future__ import print_function
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


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [
                             1, maxdisp, 1, 1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x*disp, 1)
        return out



def deconvbn_3d(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, 
                         padding=1, output_padding=1, stride=2, bias=False),
                         nn.BatchNorm3d(out_planes))


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2,
                               kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes*2))  # +conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out)+presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out)+pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet_CostVolumeAggre(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet_CostVolumeAggre, self).__init__()
        self.maxdisp = maxdisp
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        net_init(self)
        print ("[***]PSMNet_CostVolumeAggre() weights inilization done!")

    def forward(self, cost):
        """
        args:
            cost: cost volume in size [N,C,D,H,W]
        """
        print ("[???] cv shape: ", cost.shape)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, _, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, _, _ = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            # updated by CCJ: due to deprecated warning!
            #cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            #cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost1 = F.interpolate(cost1, [self.maxdisp, left.size()[2], left.size()[3]],
                                  mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp, left.size()[2], left.size()[3]],
                                  mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        #cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]],
                              mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3