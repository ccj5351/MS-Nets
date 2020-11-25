# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: main_msnet.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 07-01-2020
# @last modified: Sat 14 Mar 2020 01:51:01 AM EDT

import sys
import shutil
import os
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter
from src.dispColor import colormap_jet_batch_image,KT15FalseColorDisp,KT15LogColorDispErr

# this is cython fuction, it is SO QUICK !!!
from src.cython import writeKT15FalseColor as KT15FalseClr
from src.cython import writeKT15ErrorLogColor as KT15LogClr
import numpy as np
import time
from datetime import datetime
from PIL import Image

from src import pfmutil as pfm
from src.loss import valid_accu3, MyLoss2

# dataloading
from src.dataloader.data import get_training_set, get_iter_training_set, get_iter_testing_set
from src.dataloader.dataset import my_worker_init_fn
from src.dataloader.cbmv_generator import get_default_args_dict, generate_test_cbmv
import random

from src.funcs_utili import print_ms_gcnet_params, get_dataloader_len
#added for cbmv feature memory problem???
import psutil
import gc

import scipy.ndimage as ndimage


""" train and test MSNet """
class MyMSNet(object):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        # only generate feature for left image; otherwise, for both left and right;
        self.is_left_only = True
        print ('[---???] model_name =', self.model_name)
        self.ds = 32 # encoder sacle, that means an image with [H, W] will be downsampled up to to [H/32, W/32]; 
        if self.model_name == 'MS-GCNet':
            from src.models.gcnet_3dcnn import GCNet_CostVolumeAggre as MyAggregationModel
        elif self.model_name == 'MS-PSMNet':
            from src.models.psmnet_3dcnn import PSMNet_CostVolumeAggre as MyAggregationModel
        else:
            raise Exception("No suitable model found ...")
        self.lr = args.lr
        self.kitti2012  = args.kitti2012
        self.kitti2015  = args.kitti2015
        self.eth3d = args.eth3d
        self.middlebury = args.middlebury
        self.checkpoint_dir = args.checkpoint_dir
        self.log_summary_step = args.log_summary_step
        self.isTestingMode = (str(args.mode).lower() == 'test')
        self.cuda = args.cuda
        self.is_semantic = False
        self.max_disp = args.max_disp
        self.ms_args_dict = get_default_args_dict()
        
        if args.sf_frames is not '':
            self.ms_args_dict['sf_frames_type'] = args.sf_frames
            print ("[****] Updated sf_frames_type, new value is ", 
                    self.ms_args_dict['sf_frames_type'])
        
        self.ITER_DATALOADING = True
        if not self.isTestingMode: # training mode
            if self.ITER_DATALOADING:
                print('===> Loading Iterable-style datasets')
                self.iterable_train_set = get_iter_training_set(args.data_path, 
                        args.training_list, 
                        [args.crop_height, args.crop_width], 
                        args.kitti2012, args.kitti2015, 
                        args.eth3d, args.middlebury,
                        self.max_disp, 
                        self.is_semantic, # False, is_semantic
                        self.ms_args_dict,
                        self.is_left_only
                        )
                self.training_data_loader = DataLoader(
                        dataset = self.iterable_train_set, 
                        num_workers = args.threads, 
                        worker_init_fn = my_worker_init_fn,# newly added for ms features;
                        batch_size = args.batchSize, 
                        drop_last = True,
                        timeout = 0,
                        )
            else:
                print('===> Loading Map-style datasets')
                self.train_set = get_training_set(args.data_path, 
                        args.training_list, 
                        [args.crop_height, args.crop_width], 
                        args.kitti2012, args.kitti2015, 
                        args.eth3d, args.middlebury,
                        self.max_disp, 
                        self.is_semantic, # False, is_semantic
                        self.ms_args_dict,
                        self.is_left_only
                        )
                self.training_data_loader = DataLoader(
                        dataset = self.train_set, 
                        num_workers = args.threads, 
                        batch_size = args.batchSize, 
                        drop_last = True,
                        timeout = 0,
                        shuffle = True
                        )
            
            #assert args.threads == 0, "For ms features extraction and loading, Should be num_workers=0"
            print ("[???????????????] num_workers = %d, batch_size = %d" % (args.threads, 
                    args.batchSize))
            

            self.train_loader_len = get_dataloader_len(args.training_list, args.batchSize)
            self.criterion = MyLoss2(thresh=3, alpha=2)
        
        else:
            print('===> Loading Testing Iterable-style datasets')
            print ("[???] crop_height = %d, crop_width = %d" %(args.crop_height, args.crop_width))
            self.iterable_test_set = get_iter_testing_set(
                    args.data_path, 
                    args.test_list, 
                    [args.crop_height, args.crop_width], 
                    args.kitti2012, 
                    args.kitti2015, 
                    args.eth3d, 
                    args.middlebury,
                    self.max_disp, 
                    self.is_semantic, # False, is_semantic
                    self.ms_args_dict,
                    self.is_left_only,
                    self.ds
                    )

            self.testing_data_loader = DataLoader(
                    dataset = self.iterable_test_set, 
                    #num_workers = args.threads, 
                    num_workers = 0, 
                    worker_init_fn = my_worker_init_fn,# newly added for CBMV features;
                    batch_size = 1, 
                    drop_last = False,
                    timeout = 0,
                    )
            self.test_loader_len = get_dataloader_len(args.test_list, 1)
        
        print('===> Building {} Model'.format(self.model_name))

        if self.model_name == 'MS-GCNet' or self.model_name == 'MS-PSMNet':
            self.model = MyAggregationModel(self.max_disp, 
                        is_quarter_input_size = ( self.ms_args_dict['ds_scale'] == 4)) 
       
        else:
            raise Exception("No suitable model found ...")

        if self.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()
        
        if not self.isTestingMode: # training mode
            """ We need to set requires_grad == False to freeze the parameters 
                so that the gradients are not computed in backward();
                Parameters of newly constructed modules have requires_grad=True by default;
            """
            # updated for the cases where some subnetwork was forzen!!!
            params_to_update = [p for p in self.model.parameters() if p.requires_grad]
            if 0:
                print ('[****] params_to_update = ')
                for p in params_to_update:
                    print (type(p.data), p.size())

            print('[***]Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
            #print_ms_gcnet_params(self.model)
            #sys.exit()

            self.optimizer= optim.Adam(params_to_update, lr = args.lr, betas=(0.9,0.999))
            #self.optimizer= optim.RMSprop(params_to_update, lr = args.lr, alpha=0.9)
            self.writer = SummaryWriter(args.train_logdir)
        
        
        if self.isTestingMode:
            assert os.path.isfile(args.resume) == True, "Model Test but NO checkpoint found at {}".format(args.resume)
        if args.resume:
            if os.path.isfile(args.resume):
                print("[***] => loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                if not self.isTestingMode: # training mode
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("=> no checkpoint found at {}".format(args.resume))
        
    
    def save_checkpoint(self, epoch, state_dict, is_best=False):
        saved_checkpts = pjoin(self.checkpoint_dir, self.model_name)
        if not os.path.exists(saved_checkpts):
            os.makedirs(saved_checkpts)
            print ('makedirs {}'.format(saved_checkpts))
        
        filename = pjoin(saved_checkpts, "model_epoch_%05d.tar" % epoch)
        torch.save(state_dict, filename)
        print ('Saved checkpoint at %s' % filename) 
        if is_best:
            best_fname = pjoin(saved_checkpts, 'model_best.tar')
            shutil.copyfile(filename, best_fname)

    def adjust_learning_rate(self, epoch):
        if epoch <= 200:
            self.lr = self.args.lr
        else:
            self.lr = self.args.lr * 0.1
        
        print('learning rate = ', self.lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def load_checkpts(self, saved_checkpts = ''):
        print(" [*] Reading checkpoint %s" % saved_checkpts)
        
        checkpoint = None
        if saved_checkpts and saved_checkpts != '':
            try: #Exception Handling
                f = open(saved_checkpts, 'rb')
            except IsADirectoryError as error:
                print (error)
            else:
                checkpoint = torch.load(saved_checkpts)
        return checkpoint

    def build_train_summaries(self, imgl, imgr, disp, disp_gt, global_step, loss, epe_err, 
            is_KT15Color = False,
            #newly added for iResNet;
            disp_init = None, 
            res_disp_itr0 = None, # == disp_init + res_res_itr0
            censusL_disp = None,
            nccL_disp = None,
            sobL_disp = None,
            sadL_disp = None, 
            censusR_disp = None,
            nccR_disp = None,
            sobR_disp = None,
            sadR_disp = None, 
            cbmv_disp = None,
            ):
            """ loss and epe error """
            self.writer.add_scalar(tag = 'train_loss', scalar_value = loss, global_step = global_step)
            self.writer.add_scalar(tag = 'train_err', scalar_value = epe_err, global_step = global_step)
            """ Add batched image data to summary:
                Note: add_images(img_tensor): img_tensor could be torch.Tensor, numpy.array, or string/blobname;
                so we could use torch.Tensor or numpy.array !!!
            """
            self.writer.add_images(tag='train_imgl',img_tensor=imgl, global_step = global_step, dataformats='NCHW')
            if imgr is not None:
                self.writer.add_images(tag='train_imgr',img_tensor=imgr, global_step = global_step, dataformats='NCHW')
            
            def get_color_disp(disp):
                if disp is not None:
                    disp_tmp = KT15FalseColorDisp(disp) if is_KT15Color else colormap_jet_batch_image(disp, isGray = False)
                else:
                    disp_tmp = None
                return disp_tmp
            
            with torch.set_grad_enabled(False):
                disp_tmp = get_color_disp(disp)
                disp_gt_tmp = get_color_disp(disp_gt)
                disp_init_tmp = get_color_disp(disp_init)
                res_disp_itr0_tmp = get_color_disp(res_disp_itr0)  

                self.writer.add_images(tag='train_disp', img_tensor=disp_tmp, global_step = global_step, dataformats='NHWC')
                self.writer.add_images(tag='train_dispGT',img_tensor=disp_gt_tmp, global_step = global_step, dataformats='NHWC')
                self.writer.add_images(tag='train_dispErr',img_tensor=KT15LogColorDispErr(disp, disp_gt), 
                                       global_step = global_step, dataformats='NHWC')
                
                if disp_init is not None: 
                    self.writer.add_images(tag='train_disp_init',img_tensor=disp_init_tmp, 
                                           global_step = global_step, dataformats='NHWC')
                    self.writer.add_images(tag='train_disp_init_Err',img_tensor=KT15LogColorDispErr(disp_init, disp_gt),
                                        global_step = global_step, dataformats='NHWC')
                if res_disp_itr0 is not None: 
                    self.writer.add_images(tag='train_disp_itr0',img_tensor=res_disp_itr0_tmp, 
                                           global_step = global_step, dataformats='NHWC')
                    self.writer.add_images(tag='train_disp_itr0_Err',img_tensor=KT15LogColorDispErr(res_disp_itr0, disp_gt),
                                        global_step = global_step, dataformats='NHWC')
                for n, t in [
                    ('censusL_disp', censusL_disp),
                    ('censusR_disp', censusR_disp),
                    ('nccL_disp', nccL_disp),
                    ('nccR_disp', nccR_disp),
                    ('sobL_disp', sobL_disp),
                    ('sobR_disp', sobR_disp),
                    ('sadL_disp', sadL_disp),
                    ('sadR_disp', sadR_disp),
                    ]:
                    if t is not None:
                        # change [H/4, W/4] to [H/2, W/2]
                        t_scaled = ndimage.zoom(t[:,None,...], zoom=(1,1,2,2), order=1) # order=1, Bilinear interpolation
                        self.writer.add_images(tag='train_' + n,img_tensor= get_color_disp(t_scaled), 
                                           global_step = global_step, dataformats='NHWC')
    
                if cbmv_disp is not None: 
                    self.writer.add_images(tag='train_disp_cbmv',img_tensor= get_color_disp(cbmv_disp), 
                                           global_step = global_step, dataformats='NHWC')
                    self.writer.add_images(tag='train_disp_cbmv_Err',img_tensor=KT15LogColorDispErr(cbmv_disp, disp_gt),
                                        global_step = global_step, dataformats='NHWC')

    #---------------------
    #---- Training -------
    #---------------------
    def train(self, epoch, py = None):
        """Set up TensorBoard """
        epoch_loss = 0
        epoch_epe = 0
        epoch_accu3 = 0
        valid_iteration = 0

        
        """ shuffle the file list for data loading """
        if self.ITER_DATALOADING:
            self.iterable_train_set.random_shuffle()
        
        """ 
        print("[???????????????????] Just for debugging !!!!")
        for iteration, batch_data in enumerate(self.training_data_loader):
            print (" [***] epoch = %d, iteration = %d/%d" % (epoch, iteration, self.train_loader_len))
            cost = batch_data[0] # False by default;
            #print ("[???] input1 require_grad = ", input1.requires_grad) # False
            target = batch_data[1]
            left_rgb = batch_data[2]
            if py is not None:
                gc.collect()
                memoryUse = py.memory_info()[0] / 2.**20  # memory use in MB...I think
                message_info = 'memeory: {:.2f} MB'.format(memoryUse)
            print (message_info)
        #sys.exit()
        return -1, -1, -1
        """

        # setting to train mode;
        self.model.train()
        self.adjust_learning_rate(epoch)

        """ running log loss """
        log_running_loss = 0.0
        log_running_err  = 0.0
         

        for iteration, batch_data in enumerate(self.training_data_loader):
            start = time.time()
            #print (" [***] iteration = %d" % iteration)
            cost = batch_data[0] # False by default;
            #print ("[???] input1 require_grad = ", input1.requires_grad) # False
            target = batch_data[1]
            left_rgb = batch_data[2]
            right_rgb = batch_data[3]

            #if self.is_semantic:
            #    semantic_label = batch_data[3]
            
            if self.cuda:
                cost = cost.cuda()
                target = target.cuda()

            target = torch.squeeze(target,1)
            N,H,W = target.size()[:]
            # valid pixels: 0 < disparity < max_disp
            mask = (target - self.max_disp)*target < 0
            mask.detach_()
            valid_disp = target[mask].size()[0]
            
            if valid_disp > 0:
                self.optimizer.zero_grad()
                 
                if self.model_name == 'MS-GCNet':
                    disp = self.model(cost)
                    loss0 = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss = 0.4*loss0 + 0.6*self.criterion(disp[mask], target[mask])
                    else:
                        loss = loss0
                elif self.model_name == 'MS-PSMNet':
                    disp0, disp1, disp = self.model(cost)
                    loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                    loss1 = F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean')
                    if self.kitti2012 or self.kitti2015:
                        loss2 = self.criterion(disp[mask], target[mask])
                    else:
                        loss2 = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
                    
                    loss = 0.2*loss0 + 0.6*loss1 + loss2
                
                else:
                    raise Exception("No suitable model found ... Wrong name : {}".format(self.model_name))
                
                loss.backward()
                self.optimizer.step()
                # MAE error
                error = torch.mean(torch.abs(disp[mask] - target[mask]))
                # accu3 
                accu_thred = 3.0
                accu = valid_accu3(target[mask], disp[mask], thred=accu_thred)

                epoch_loss += loss.item()
                epoch_epe += error.item()
                epoch_accu3 += accu.item() 
                valid_iteration += 1
                

                # epoch - 1: here argument `epoch` is starting from 1, instead of 0 (zer0);
                train_global_step = (epoch-1)*self.train_loader_len + iteration      
                message_info = "===> Epoch[{}]({}/{}): Step {}, Loss: {:.3f}, EPE: {:.2f}, Acu{:.1f}: {:.2f}; {:.2f} s/step".format(
                                epoch, iteration, self.train_loader_len, train_global_step,
                                loss.item(), error.item(), accu_thred, accu.item(), time.time() - start)
                """ adapted from Mateo's code """
                if py is not None:
                    gc.collect()
                    memoryUse = py.memory_info()[0] / 2.**20  # memory use in MB...I think
                    message_info += ', memeory: {:.2f} MB'.format(memoryUse)
                print (message_info)
                sys.stdout.flush()

                # save summary for tensorboard visualization
                log_running_loss += loss.item()
                log_running_err += error.item()

                
                #For tensorboard visulization, we could just show half size version, i.e., [H/2, W/2], 
                if iteration % self.log_summary_step == (self.log_summary_step - 1):
                    dsi = cost.cpu().numpy()
                    cenL_disp = np.argmin(dsi[:,0,:,:,:],axis=1).astype(np.float32)
                    nccL_disp = np.argmin(dsi[:,1,:,:,:],axis=1).astype(np.float32)
                    sobL_disp = np.argmin(dsi[:,2,:,:,:],axis=1).astype(np.float32)
                    sadL_disp = np.argmin(dsi[:,3,:,:,:],axis=1).astype(np.float32)
                    
                    cenR_disp=None
                    nccR_disp=None
                    sobR_disp=None
                    sadR_disp=None
                    if not self.is_left_only:
                        cenR_disp = np.argmin(dsi[:,8,:,:,:],axis=1).astype(np.float32)
                        nccR_disp = np.argmin(dsi[:,9,:,:,:],axis=1).astype(np.float32)
                        sobR_disp = np.argmin(dsi[:,10,:,:,:],axis=1).astype(np.float32)
                        sadR_disp = np.argmin(dsi[:,11,:,:,:],axis=1).astype(np.float32)
                    
                    self.build_train_summaries( 
                          #left_rgb, 
                          F.interpolate(left_rgb, size=[H//2, W//2], mode='bilinear', align_corners = True),
                          F.interpolate(right_rgb, size=[H//2, W//2], mode='bilinear', align_corners = True),
                          #None, #right_rgb,
                          # in the latest versions of PyTorch you can add a new axis by indexing with None 
                          # > see: https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155;
                          #torch.unsqueeze(disp, dim=1) ==> disp[:,None]
                          F.interpolate(disp[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True), 
                          F.interpolate(target[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True),
                          #disp[:,None], target[:,None],
                          train_global_step, 
                          log_running_loss/self.log_summary_step, 
                          log_running_err/self.log_summary_step, 
                          is_KT15Color = False,
                          #is_KT15Color = True,
                          # from disp_results['res_disp0_itr0']
                          disp_init = F.interpolate(disp0[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True) if self.model_name == 'CBMV-iResNet' else None,
                          #from disp_results['res_disp0_itr0'], # == disp_init + res_res_itr0
                          res_disp_itr0 = F.interpolate(disp1[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True) if self.model_name == 'CBMV-iResNet' else None,
                          
                          # visualize disparity inferred from cbmv features via argmin()
                          censusL_disp = cenL_disp,
                          nccL_disp = nccL_disp,
                          sobL_disp = sobL_disp,
                          sadL_disp = sadL_disp, 
                          censusR_disp = cenR_disp,
                          nccR_disp = nccR_disp,
                          sobR_disp = sobR_disp,
                          sadR_disp = sadR_disp,
                          cbmv_disp = F.interpolate(disp_cbmv[:,None,...], size=[H//2, W//2], mode='bilinear', align_corners = True) if self.model_name == 'CBMV-DispNetC-V3' else None
                        )
                    # reset to zeros
                    log_running_loss = 0.0
                    log_running_err = 0.0
                
                # about memory leaking 
                del cost

        
        # end of data_loader
        # save the checkpoints
        avg_loss = epoch_loss / valid_iteration
        avg_err = epoch_epe / valid_iteration
        avg_accu = epoch_accu3 / valid_iteration
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. EPE Error: {:.4f}, Accu{:.1f}: {:.4f})".format(
                  epoch, avg_loss, avg_err, accu_thred, avg_accu))

        is_best = False
        model_state_dict = {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'loss': avg_loss,
                        'epe_err': avg_err, 
                        'accu3': avg_accu
                    }

        if self.kitti2012 or self.kitti2015:
            #if epoch % 50 == 0 and epoch >= 300:
            #if epoch % 50 == 0:
            if epoch % 25 == 0:
                self.save_checkpoint(epoch, model_state_dict, is_best)
        else:
            #if epoch >= 7:
            #    self.save_checkpoint(epoch, model_state_dict, is_best)
            self.save_checkpoint(epoch, model_state_dict, is_best)
        # avg loss
        return avg_loss, avg_err, avg_accu
    
    #---------------------
    #---- Test -----------
    #---------------------
    def test(self, py= None):
        self.model.eval()
        if not os.path.exists(self.args.resultDir):
            os.makedirs(self.args.resultDir)
            print ('makedirs {}'.format(self.args.resultDir))
        dispScale = 1.0
        avg_err = .0
        avg_rate = .0
        img_num = self.test_loader_len

        print ("[****] To test %d images !!!" %img_num)
        for iteration, batch_data in enumerate(self.testing_data_loader):
            batch_size = batch_data[0].size()[0] 
            assert batch_size == 1
            features = batch_data[0] 
            print ("[???] features size:", features.size())
            height = batch_data[1][0].item()
            width = batch_data[2][0].item()
            crop_height = batch_data[3][0].item()
            crop_width = batch_data[4][0].item()
            current_file = batch_data[5][0]
            disp_name = batch_data[6][0]
            print (height, crop_height, current_file, disp_name)
            
            if os.path.isfile(disp_name):
                dispGT = pfm.readPFM(disp_name)
                dispGT[dispGT == np.inf] = .0
            else:
                dispGT= None
            if self.kitti2015 or self.kitti2012:
                savename = pjoin(self.args.resultDir, current_file[0:-4] + '.pfm')
                
            elif self.eth3d or self.middlebury:
                savename = pjoin(self.args.resultDir, current_file + '.pfm')
            
            else: # scene flow dataset
                savename = pjoin(self.args.resultDir, str(iteration) + '.pfm')
            
            if self.cuda:
                features = features.cuda()
            with torch.no_grad():
                if self.model_name == 'MS-GCNet':
                    disp = self.model(features)
                elif self.model_name == 'MS-PSMNet':
                    disp = self.model(features)
              
                else:
                    raise Exception("No suitable model found ... Wrong name: {}".format(self.model_name))
            #about memory
            del features 
            
            disp = disp.cpu().detach().numpy()*dispScale
            if height <= crop_height and width <= crop_width:
                #disp = disp[0, crop_height - height: crop_height, crop_width-width: crop_width]
                disp = disp[0, crop_height - height: crop_height, 0:width]
            else:
                disp = disp[0, :, :]
            
            #save to uint16 png files
            #skimage.io.imsave(savename, (disp * 256).astype('uint16'))
            if any([self.kitti2015, self.kitti2012, self.eth3d, self.middlebury,iteration%50 == 0]):
                pfm.save(savename, disp)
                #print ('saved ', savename)
            
            if dispGT is not None:
                if self.eth3d:
                    self.args.threshold = 1.0
                elif self.middlebury:
                    self.args.threshold = 1.0 # for trainingH;
                elif self.kitti2012 or self.kitti2015:
                    self.args.threshold = 3.0
                else: # Scene Flow
                    self.args.threshold = 1.0

                error, rate = get_epe_rate(dispGT, disp, self.max_disp, self.args.threshold)
                avg_err += error
                avg_rate += rate
                if iteration % 200 == 0:
                    message_info = "===> Frame {}: ".format(iteration) + current_file + " ==> EPE Error: {:.4f}, Bad-{:.1f} Error: {:.4f}".format(
                        error, self.args.threshold, rate)
                    """ adapted from Mateo's code """
                    if py is not None:
                        gc.collect()
                        memoryUse = py.memory_info()[0] / 2.**20  # memory use in MB...I think
                        message_info += ', memeory: {:.2f} MB'.format(memoryUse)
                    print (message_info)
            

            # save kt15 color
            #if self.kitti2015:
            if any([self.kitti2015, self.kitti2012, self.eth3d, self.middlebury]):
                """ disp """
                tmp_dir = pjoin(self.args.resultDir, "dispColor")
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                tmp_dispname = pjoin(tmp_dir, current_file[0:-4] + '.png')
                cv2.imwrite(tmp_dispname, 
                        KT15FalseClr.writeKT15FalseColor(np.ascontiguousarray(disp)).astype(np.uint8)[:,:,::-1])
                if iteration % 50 == 0:
                    print ('saved ', tmp_dispname)
                if dispGT is not None: #If KT benchmark submission, then No dispGT;
                    """ err-disp """
                    tmp_dir = pjoin(self.args.resultDir, "errDispColor")
                    if not os.path.exists(tmp_dir):
                        os.makedirs(tmp_dir)
                    tmp_errdispname = pjoin(tmp_dir, current_file[0:-4]  + '.png')
                    cv2.imwrite(tmp_errdispname, 
                            KT15LogClr.writeKT15ErrorDispLogColor(np.ascontiguousarray(disp), np.ascontiguousarray(dispGT)).astype(np.uint8)[:,:,::-1])
                    if iteration % 50 == 0:
                        print ('saved ', tmp_errdispname)
        if dispGT is not None:
            avg_err = avg_err / img_num
            avg_rate = avg_rate / img_num
            print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
                img_num, avg_err, self.args.threshold, avg_rate))
        print ('{} testing finished!'.format(self.model_name))
                


    #-------------------------------------
    #---- Computer Bad-X Error -----------
    #-------------------------------------
    def eval_bad_x(self):
        self.model.eval()
        file_path = self.args.data_path
        file_list = self.args.test_list
        f = open(file_list, 'r')
        filelist = [l.rstrip() for l in f.readlines()]
        avg_err = .0
        avg_rate = .0

        if not os.path.exists(self.args.resultDir):
            os.makedirs(self.args.resultDir)
            print ('makedirs {}'.format(self.args.resultDir))
        
        print ("[***]To test %d imgs" %len(filelist))
        for index in range(len(filelist)):
            current_file = filelist[index]
            if self.kitti2015:
                dispname = pjoin(file_path, 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm')
                dispGT = pfm.readPFM(dispname)
                dispGT[dispGT == np.inf] = .0
                savename = pjoin(self.args.resultDir, current_file[0:-4] + '.pfm')
                if os.path.isfile(savename):
                    disp = pfm.readPFM(savename)
                else:
                    savename = pjoin(self.args.resultDir, "disp-pfm/" + current_file[0:-4] + '.pfm')
                    disp = pfm.readPFM(savename)
                
            elif self.kitti2012:
                dispname = pjoin(file_path, 'disp_occ_0_pfm/' + current_file[0:-4] + '.pfm')
                dispGT = pfm.readPFM(dispname)
                dispGT[dispGT == np.inf] = .0
                savename = pjoin(self.args.resultDir, current_file[0:-4] + '.pfm')
                if os.path.isfile(savename):
                    disp = pfm.readPFM(savename)
                else:
                    savename = pjoin(self.args.resultDir, "disp-pfm/" + current_file[0:-4] + '.pfm')
                    disp = pfm.readPFM(savename)

            else:
                raise Exception("No KT dataset found, so do nothing!!!")
            

            error, rate = get_epe_rate(dispGT, disp, self.max_disp, self.args.threshold)
            avg_err += error
            avg_rate += rate
            

        avg_err = avg_err / len(filelist)
        avg_rate = avg_rate / len(filelist)
        print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Bad-{:.1f} Error: {:.4f}".format(
            len(filelist), avg_err, self.args.threshold, avg_rate))
        print ('{} testing finished!'.format(self.model_name))

def get_epe_rate(disp, prediction, max_disp = 192, threshold = 3.0):
    mask = np.logical_and(disp >= 0.001, disp <= max_disp)
    error = np.mean(np.abs(prediction[mask] - disp[mask]))
    rate = np.sum(np.abs(prediction[mask] - disp[mask]) > threshold) / np.sum(mask)
    #print(" ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    return error, rate


def main(args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
     

    #----------------------------
    # some initilization thing 
    #---------------------------
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    if args.seed > 0:
        random.seed(args.seed)
    else:
        print ("[****] set random seed as time now!!!")
        random.seed(datetime.now())
    if cuda:
        torch.cuda.manual_seed(args.seed)
    
    myNet = MyMSNet(args)
    
    print('Number of {} model parameters: {}'.format(args.model_name,
            sum([p.data.nelement() for p in myNet.model.parameters()])))
    if 0: 
        print('Including:\n1) number of Feature Extraction module parameters: {}'.format(
            sum(
                [p.data.nelement() for n, p in myNet.model.named_parameters() if any(
                    ['module.convbn0' in n, 
                     'module.res_block' in n, 
                     'module.conv1' in n
                     ])]
                )))
        print('2) number of Other modules parameters: {}'.format(
            sum(
                [p.data.nelement() for n, p in myNet.model.named_parameters() if any(
                    ['module.conv3dbn' in n,
                     'module.block_3d' in n,
                     'module.deconv' in n,
                     ])]
                )))

        for i, (n, p) in enumerate(myNet.model.named_parameters()):
            print (i, "  layer ", n, "has # param : ", p.data.nelement())
        #sys.exit()

    if args.mode == 'train':
        print('strat training !!!')
        for epoch in range(1 + args.startEpoch, args.startEpoch + args.nEpochs + 1):
            print ("[**] do training at epoch %d/%d" % (epoch, args.startEpoch + args.nEpochs))
            
            with torch.autograd.set_detect_anomaly(True):
                # about memory leaking
                pid = os.getpid()
                py = psutil.Process(pid)
                avg_loss, avg_err, avg_accu = myNet.train(epoch, py)
        # save the last epoch always!!
        myNet.save_checkpoint(args.nEpochs + args.startEpoch,
            {
                'epoch': args.nEpochs + args.startEpoch,
                'state_dict': myNet.model.state_dict(),
                'optimizer' : myNet.optimizer.state_dict(),
                'loss': avg_loss,
                'epe_err': avg_err, 
                'accu3': avg_accu
            }, 
            is_best = False)
        print('done training !!!')
    
    elif args.mode == 'test': 
        #dispname = './results/000001_10.pfm'
        #disp = pfm.readPFM(dispname)
        #tmp_dispname = './results/000001_10_disp_color.png'
        #cv2.imwrite(tmp_dispname, KT15FalseClr.writeKT15FalseColor(np.ascontiguousarray(disp)).astype(np.uint8)[:,:,::-1])
        #sys.exit()
        print('strat testing !!!')
        # about memory leaking
        pid = os.getpid()
        py = psutil.Process(pid)
        myNet.test(py)
    elif args.mode == 'eval-badx':
        myNet.eval_bad_x()



if __name__ == '__main__':
    
    import argparse
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GANet Example')
    parser.add_argument('--crop_height', type=int, required=True, help="crop height")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp")
    parser.add_argument('--crop_width', type=int, required=True, help="crop width")
    parser.add_argument('--resume', type=str, default='', help="resume from saved model")
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--log_summary_step', type=int, default=200, help='every 200 steps to build training summary')
    parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--startEpoch', type=int, default=0, help='starting point, used for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--seed',  type=int, default=-1, help='random seed to use. Default=123')
    parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
    parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012 dataset? Default=False')
    parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    parser.add_argument('--eth3d', type=int, default=0, help='eth2d dataset? Default=False')
    parser.add_argument('--middlebury', type=int, default=0, help='middlebury dataset? Default=False')
    parser.add_argument('--data_path', type=str, default='/data/ccjData', help="data root")
    parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
    parser.add_argument('--test_list', type=str, default='./lists/sceneflow_test_select.list', help="evaluation list")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help="location to save models")
    parser.add_argument('--train_logdir', dest='train_logdir',  default='./logs/tmp', help='log dir')
    """Arguments related to run mode"""
    parser.add_argument('--model_name', type=str, default='MS-GCNet', help="model name")
    parser.add_argument('--mode', dest='mode', type = str, default='train', help='train, test')
    parser.add_argument('--resultDir', type=str, default= "./results")
    parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
    """ added for Scene Flow frames type """
    parser.add_argument('--sf_frames', dest='sf_frames', type = str, default='frames_cleanpass', help='frames_cleanpass or frames_finalpass')

    args = parser.parse_args()
    print('[***] args = ', args)
    main(args)
