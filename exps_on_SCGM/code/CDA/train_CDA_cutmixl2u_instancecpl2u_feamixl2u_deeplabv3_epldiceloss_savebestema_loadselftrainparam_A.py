#!!!关于covid19_cda------
#!!!import argparse
import logging
import os
import random
#!!!import shutil
import sys
#!!!import time

import numpy as np
import torch
#!!!import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
#!!!from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
# lossfrom torchvision.utils import make_grid
from tqdm import tqdm
#!!!from itertools import cycle
import numpy as np
import cv2 

#!!!from dataloaders import utils
#!!!from dataloaders.dataset_covid import (CovidDataSets, RandomGenerator)
#!!!from networks.net_factory import net_factory
#!!!from utils import losses #!!!, metrics, ramps
#!!!from test_covid import get_model_metric
#!!!from models.pix2pix_model import Pix2PixModel, get_opt

#!!!import os.path as osp
from networks.ema import ModelEMA
from utils.transform import obtain_cutmix_box,mix
#!!!import torchvision.utils as vutils
#!!!from test_covid import get_model_metric
# from utils.tumor_cp import tumor_cp_augmentation_transformV2
#!!!---------


#!!!关于EPL------
import os
import math
from torch.utils.data import ConcatDataset #!!!Dataset, ConcatDataset
import torch.distributed as dist
#!!!import torchvision.models as models
from util.utils import get_device
from scgm_dataloader import get_meta_split_data_loaders
from config_scgm_cutmixinstancefeamix_deeplabv3_epldiceloss_loadselftrainparam_A import default_config
from util.dice_loss import dice_coeff
import util.mask_gen as mask_gen
from util.custom_collate import SegCollate
from networks.scgm_network import my_net
from utils.copy_paste import CopyPaste
CP = CopyPaste()
#!!!------

from utils.fixseed import set_random_seed
set_random_seed(14)


config = default_config
gpus = config['gpus']
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
device = get_device()


def obtain_bbox(batch_size, img_size):
    for i in range(batch_size):  
        if i == 0:
            MixMask = obtain_cutmix_box(img_size).unsqueeze(0)
        else:
            MixMask = torch.cat((MixMask, obtain_cutmix_box(img_size).unsqueeze(0)))
    return MixMask

def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.1  # 1e-12

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    #A_sum = torch.sum(tflat * iflat)
    #B_sum = torch.sum(tflat * tflat)
    loss = ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss


def pre_data(batch_size, num_workers, test_vendor):
    test_vendor = test_vendor
    
    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(
            test_vendor=test_vendor, config=config)

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    label_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    unlabel_dataset = ConcatDataset(
        [domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset])
    # unlabel_dataset = domain_2_unlabeled_dataset

    print("before length of label_dataset", len(label_dataset))

    new_labeldata_num = len(unlabel_dataset) // len(label_dataset) + 1
    new_label_dataset = label_dataset
    for i in range(new_labeldata_num):
        new_label_dataset = ConcatDataset([new_label_dataset, label_dataset])
    label_dataset = new_label_dataset
    

    # For CutMix
    mask_generator = mask_gen.BoxMaskGenerator(prop_range=config['cutmix_mask_prop_range'], n_boxes=config['cutmix_boxmask_n_boxes'],
                                               random_aspect_ratio=config['cutmix_boxmask_fixed_aspect_ratio'],
                                               prop_by_area=config['cutmix_boxmask_by_size'], within_bounds=config[
                                                   'cutmix_boxmask_outside_bounds'],
                                               invert=config['cutmix_boxmask_no_invert'])

    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    label_loader = DataLoader(dataset=label_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    unlabel_loader_0 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=mask_collate_fn)

    unlabel_loader_1 = DataLoader(dataset=unlabel_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=False, collate_fn=collate_fn)

    print("after length of label_dataset", len(label_dataset))
    print("length of unlabel_dataset", len(unlabel_dataset))
    print("length of val_dataset", len(val_dataset))
    print("length of test_dataset", len(test_dataset))

    return label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, len(label_dataset), len(unlabel_dataset)



def train(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, learning_rate, weight_decay, num_epoch, model_path, niters_per_epoch):

    # Initialize model
    model1 = my_net(modelname='mydeeplabV3P', default_config=default_config)    
    if default_config['Loadselftrain']:
        model_dict = model1.state_dict()
        pretrained_dict = torch.load(default_config['pretrain_file']) 
        pretrained_dict = pretrained_dict.state_dict()
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model1.load_state_dict(model_dict)
        print('load {}'.format(default_config['pretrain_file']))    
    model1 = model1.to(device)
    model1.device = device
    model1 = nn.DataParallel(model1, device_ids=gpus, output_device=gpus[0])
    
    ema_model = ModelEMA(model1, 0.9) #!!!0.7)#!!! 0.8) #!!!0.999)
    teacher_model = ema_model.ema    
    teacher_model = teacher_model.to(device)
    teacher_model.device = device
    teacher_model = nn.DataParallel(teacher_model, device_ids=gpus, output_device=gpus[0])
    
    # Initialize optimizer.
    # optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    iter_num = 0
    max_iterations = num_epoch * len(label_loader)
    best_dice = 0
    best_dice_ema = 0
    best_dice_test = 0
    best_dice_test_ema = 0
    
    cross_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    
    savedir='./tmodel_scgm/1gpu/cutmixl2u_instancecpl2u_feamixl2u_deeplabv3_epldiceloss_savebestema_loadselftrainparam_A/nousepseudoloss_lr{}wdecay{}_alpha0.4_ema0.9_seed14/'.format(config['learning_rate'], config['weight_decay'])
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    writer = SummaryWriter(savedir + '/log_{}'.format(config['test_vendor']))
        
    for epoch in range(num_epoch):
        # ---------- Training ----------
        model1.train()
        teacher_model.train()
        
        label_dataloader = iter(label_loader)
        unlabel_dataloader_0 = iter(unlabel_loader_0)

        # loss data
        epoch_loss = []
        epoch_loss_labeled = []
        epoch_loss_pseudo = []
        epoch_loss_pseudo_aug = []
        
        # tqdm
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        
        for idx in pbar:
            minibatch = label_dataloader.next()
            imgs = minibatch['img']#!!! shape:[b,1,288,288]
            mask = minibatch['mask']#!!! shape:[b,2,288,288]
            imgs = imgs.to(device)
            mask = mask.to(device=device, dtype=torch.long)
            
            unsup_minibatch_0 = unlabel_dataloader_0.next()
            unsup_imgs_0 = unsup_minibatch_0['img']
            unsup_imgs_0 = unsup_imgs_0.to(device)
            
            # get pseudo labels from teacher_model for unlabeled data
            teacher_model.eval()
            with torch.no_grad():
                outputs_unlabeled,_ = teacher_model(unsup_imgs_0)
                outputs_unlabeled_soft = torch.softmax(outputs_unlabeled, dim=1)
                pseudo_labels = torch.argmax(outputs_unlabeled_soft.detach(), dim=1, keepdim=False)
                
            teacher_model.train()
            
            # Strong view: Cross-set data augmentation (cutmix) 
            img_size = 288 
            bs1 = imgs.shape[0]
            bs2 = unsup_imgs_0.shape[0]
            MixMask = obtain_bbox(bs1+bs2, img_size).cuda()
            # print('MixMask1.shape, input_l.shape, input_ul.shape:',MixMask1.shape, input_l.shape, input_ul.shape)
            input_aug1,rand_index1 = mix(MixMask[:bs1].unsqueeze(1).repeat(1, 1, 1, 1), imgs, unsup_imgs_0) #l-->ul
            # input_aug2,rand_index2 = mix(MixMask[bs1:bs1+bs2].unsqueeze(1).repeat(1, 1, 1, 1), unsup_imgs_0, imgs) #ul-->l
            pseudo_labels_aug1,_ = mix(MixMask[:bs1], mask[:,1,:,:], pseudo_labels, rand_index1)
            # pseudo_labels_aug2,_ = mix(MixMask[bs1:bs1+bs2], pseudo_labels, mask[:,1,:,:], rand_index2)
            
            # Strong view: Cross-set data augmentation (copy-paste)
            #!!!l2u
            pseudo_labels_new = 1- pseudo_labels
            # torchvision.utils.save_image(pseudo_labels.unsqueeze(1).float(),'pseudo_labels2_{}.png'.format(epoch))
            pseudo_imgs_l2u, pseudo_mask_l2u, _, _ = CP(unsup_imgs_0, pseudo_labels_new, imgs, mask[:,0,:,:])
            
            '''
            import torchvision
            torchvision.utils.save_image(imgs,'imgs_{}.png'.format(epoch))
            torchvision.utils.save_image(mask[:,0,:,:].unsqueeze(1).float(),'mask_{}.png'.format(epoch))
            torchvision.utils.save_image(unsup_imgs_0,'unsup_imgs_0_{}.png'.format(epoch))
            torchvision.utils.save_image(pseudo_labels.unsqueeze(1).float(),'pseudo_labels_{}.png'.format(epoch))
            torchvision.utils.save_image(input_aug1,'input_aug1_{}.png'.format(epoch))
            torchvision.utils.save_image(input_aug2,'input_aug2_{}.png'.format(epoch))
            torchvision.utils.save_image(pseudo_labels_aug1.unsqueeze(1).float(),'pseudo_labels_aug1_{}.png'.format(epoch))
            torchvision.utils.save_image(pseudo_labels_aug2.unsqueeze(1).float(),'pseudo_labels_aug2_{}.png'.format(epoch))
            if epoch > 3:
                print(kiki)
            '''
        
            '''
            volume_batch = torch.cat([imgs, unsup_imgs_0, input_aug1, pseudo_imgs_l2u], 0)
            label_batch = torch.cat([mask[:,0,:,:], pseudo_labels, pseudo_labels_aug1, pseudo_mask_l2u], 0)
            '''
            #!!!
            #!!!feature-level mixup l->u
            in1 = torch.cat((imgs,unsup_imgs_0))
            mix_alpha=0.4 #1.0 #0.9 #!!0.4
            lambda1 = np.random.beta(mix_alpha, mix_alpha) #!!!lambda1 = 0.5 #!!!np.random.beta(mix_alpha, mix_alpha)
            # lambda1 = 0.5
            pred = model1(in1, need_feamix=True, lambda1=lambda1)
            _, _, pre_feamix_1 = pred.chunk(3)
            outputs_soft_feamix_1 = torch.softmax(pre_feamix_1, dim=1)
            #!!!
            
            volume_batch = torch.cat([imgs, unsup_imgs_0, input_aug1, pseudo_imgs_l2u], 0)
            label_batch = torch.cat([mask[:,0,:,:], pseudo_labels, pseudo_labels_aug1, pseudo_mask_l2u], 0)
            
            outputs, _ = model1(volume_batch) #!!! shape:[b,2,288,288]
            outputs_soft = torch.softmax(outputs, dim=1)
            
            labeled_loss_1 = dice_loss(outputs_soft[:config['batch_size']][:,0,:,:], label_batch[:config['batch_size']])
            pseudo_supervision = dice_loss(outputs_soft[config['batch_size']:config['batch_size']*2][:,1,:,:], label_batch[config['batch_size']:config['batch_size']*2])
            pseudo_supervision_aug1 = dice_loss(outputs_soft[config['batch_size']*2:config['batch_size']*3][:,1,:,:], label_batch[config['batch_size']*2:config['batch_size']*3])
            pseudo_supervision_aug2 = dice_loss(outputs_soft[config['batch_size']*3:config['batch_size']*4][:,0,:,:], label_batch[config['batch_size']*3:config['batch_size']*4])
            pseudo_supervision_aug3 = lambda1*dice_loss(outputs_soft_feamix_1[:,0,:,:], mask[:,0,:,:]) + (1-lambda1)*dice_loss(outputs_soft_feamix_1[:,0,:,:], 1-pseudo_labels) #！！+ (1-lambda1)*dice_loss(outputs_soft_feamix_1[:,1,:,:], pseudo_labels)
            
            if epoch <= 3:
                loss = labeled_loss_1
            else:
                loss = labeled_loss_1 + pseudo_supervision_aug1+pseudo_supervision_aug2 + pseudo_supervision_aug3 #!!!+pseudo_supervision + pseudo_supervision_aug1+pseudo_supervision_aug2 + pseudo_supervision_aug3   #!!!+ pseudo_supervision_aug1+pseudo_supervision_aug2 + pseudo_supervision_aug3 #!!!pseudo_supervision + pseudo_supervision_aug1+pseudo_supervision_aug2 + pseudo_supervision_aug3   
            epoch_loss.append(float(loss))
            epoch_loss_labeled.append(float(labeled_loss_1))
            #!!!epoch_loss_pseudo.append(float(pseudo_supervision))
            epoch_loss_pseudo_aug.append(float((pseudo_supervision_aug1+pseudo_supervision_aug2+pseudo_supervision_aug3)/3.0))
            

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            
            ema_model.update(model1)
            teacher_model = ema_model.ema

            
            # lr schedule
            iter_num = iter_num + 1
            '''
            lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            '''
            
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_loss_labeled = sum(epoch_loss_labeled) / len(epoch_loss_labeled)
        #!!!epoch_loss_pseudo = sum(epoch_loss_pseudo) / len(epoch_loss_pseudo)
        epoch_loss_pseudo_aug = sum(epoch_loss_pseudo_aug) / len(epoch_loss_pseudo_aug)
        
        # Print the information.
        print(
            f"[ Normal image Train | {epoch + 1:03d}/{num_epoch:03d} ] lr = {learning_rate:.4}, epoch_loss = {epoch_loss:.5f}, epoch_loss_l = {epoch_loss_labeled:.5f}, epoch_loss_u_aug = {epoch_loss_pseudo_aug:.5f}")
        
        # ---------- Validation ----------
        val_loss1, val_dice1 = test(model1, val_loader)
        val_loss1_ema, val_dice1_ema = test(teacher_model, val_loader)
        print(
            f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] val_loss1 = {val_loss1:.5f} val_dice1 = {val_dice1:.5f} val_loss1_ema = {val_loss1_ema:.5f} val_dice1_ema = {val_dice1_ema:.5f}")

        # ---------- Testing (using ensemble)----------
        test_loss1, test_dice1= test(model1, test_loader)
        test_loss1_ema, test_dice1_ema= test(teacher_model, test_loader)
        print(
            f"[ Test | {epoch + 1:03d}/{num_epoch:03d} ] test_loss1 = {test_loss1:.5f} test_dice1 = {test_dice1:.5f} test_loss1_ema = {test_loss1_ema:.5f} test_dice1_ema = {test_dice1_ema:.5f}")
        
        # val
        text = {'val/val_dice1': val_dice1}
        print(text)
        text = {'val/val_dice1_ema': val_dice1_ema}
        print(text)
        # test
        text = {'test/test_dice1': test_dice1}
        print(text)
        text = {'test/test_dice1_ema': test_dice1_ema}
        print(text)
        # loss
        text = {'epoch': epoch + 1, 'loss/epoch_loss': epoch_loss, 'loss/epoch_loss_l': epoch_loss_labeled, #!!!'loss/epoch_loss_u': epoch_loss_pseudo,
                   'loss/test_loss1': test_loss1, 'loss/val_loss1': val_loss1,
                   'loss/test_loss1_ema': test_loss1_ema, 'loss/val_loss1_ema': val_loss1_ema}
        print(text)

        # if the model improves, save a checkpoint at this epoch
        if val_dice1 > best_dice:
            best_dice = val_dice1
            # 使用了多GPU需要加上module
            print('saving model with best_dice {:.5f}'.format(best_dice))
            model_name = savedir + 'stu_'+ model_path
            torch.save(model1.module, model_name)
        if val_dice1_ema > best_dice_ema:
            best_dice_ema = val_dice1_ema
            # 使用了多GPU需要加上module
            print('saving model with best_dice {:.5f}'.format(best_dice_ema))
            model_name = savedir + 'tea_'+ model_path
            torch.save(teacher_model.module, model_name)
        if test_dice1 > best_dice_test:
            best_dice_test = test_dice1
            # 使用了多GPU需要加上module
            print('saving model with best_dice {:.5f}'.format(best_dice_test))
            model_name = savedir + 'stu_test_'+ model_path
            torch.save(model1.module, model_name)
        if test_dice1_ema > best_dice_test_ema:
            best_dice_test_ema = test_dice1_ema
            # 使用了多GPU需要加上module
            print('saving model with best_dice {:.5f}'.format(best_dice_test_ema))
            model_name = savedir + 'tea_test_'+ model_path
            torch.save(teacher_model.module, model_name)
            
                        
        writer.add_scalar('lr', optimizer1.param_groups[0]['lr'], iter_num)
        writer.add_scalar('loss/epoch_loss', epoch_loss, iter_num)
        writer.add_scalar('loss/epoch_loss_l', epoch_loss_labeled, iter_num)
        #!!!writer.add_scalar('loss/epoch_loss_u', epoch_loss_pseudo, iter_num)
        writer.add_scalar('loss/val_loss1', val_loss1, iter_num)
        writer.add_scalar('loss/test_loss1', test_loss1, iter_num)
        writer.add_scalar('loss/val_dice1', val_dice1, iter_num)
        writer.add_scalar('loss/test_dice1', test_dice1, iter_num)
        writer.add_scalar('loss/val_dice1_ema', val_dice1_ema, iter_num)
        writer.add_scalar('loss/test_dice1_ema', test_dice1_ema, iter_num)
            
    writer.close()

''''            
# use the function to calculate the valid loss or test loss
def test_dual(model_l, model_r, loader):
    model_l.eval()
    if not model_r is None:
        model_r.eval()
    
    loss_l = []
    loss_r = []
    t_loss = 0
    r_loss = 0

    tot = 0

    for batch in tqdm(loader):
        imgs = batch['img']
        mask = batch['mask']
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits_l, _ = model_l(imgs)
            sof_l = F.softmax(logits_l, dim=1)
            pred = sof_l
            if not model_r is None:
                logits_r, _ = model_r(imgs)
                sof_r = F.softmax(logits_r, dim=1)
                pred = (sof_l + sof_r) / 2
                
        pred = (pred > 0.5).float()

        # loss
        t_loss = dice_loss(pred[:, 0, :, :], mask[:, 0, :, :])
        loss_l.append(t_loss.item())

        # dice score
        tot += dice_coeff(pred[:, 0, :, :],mask[:, 0, :, :], device).item()

    loss_l = sum(loss_l) / len(loss_l)

    dice = tot/len(loader)

    model_l.train()
    if not model_r is None:
        model_r.train()
        
    return loss_l, dice
'''

def test(model, loader):
    model.eval()
    
    loss = []
    t_loss = 0

    tot = 0

    for batch in tqdm(loader):
        imgs = batch['img']
        mask = batch['mask']
        imgs = imgs.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits, _ = model(imgs)
            pred = F.softmax(logits, dim=1)
                
        pred = (pred > 0.5).float()

        # loss
        t_loss = dice_loss(pred[:, 0, :, :], mask[:, 0, :, :])
        loss.append(t_loss.item())

        # dice score
        tot += dice_coeff(pred[:, 0, :, :],mask[:, 0, :, :], device).item()

    loss = sum(loss) / len(loss)

    dice = tot/len(loader)

    model.train()
        
    return loss, dice


if __name__ == '__main__':
    print('------------------------------')
    print(config)
    print('------------------------------')
    
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epoch = config['num_epoch']
    model_path = config['model_path']
    test_vendor = config['test_vendor']

    label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, num_label_imgs, num_unsup_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)

    max_samples = num_unsup_imgs
    niters_per_epoch = int(math.ceil(max_samples * 1.0 // batch_size))
    print("max_samples", max_samples)
    print("niters_per_epoch", niters_per_epoch)

    if config['Fourier_aug']:
        print("Fourier mode")
    else:
        print("Normal mode")

    train(label_loader, unlabel_loader_0, unlabel_loader_1, test_loader, val_loader, learning_rate,
          weight_decay, num_epoch, model_path, niters_per_epoch)