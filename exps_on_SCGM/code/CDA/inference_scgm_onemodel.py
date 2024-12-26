import os
#!!!os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import math
import statistics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

#!!!from network.scgm_network import my_net
from util.utils import get_device, dice_loss, im_convert, label_to_onehot
from scgm_dataloader import get_meta_split_data_loaders
from config_scgm_deeplabv3_epldiceloss_A import default_config #!!!from config import default_config
from util.data_utils import save_image
from util.dice_loss import dice_coeff
#!!!from draw_dataloader import OneImageFolder


from utils.fixseed import set_random_seed
set_random_seed(14)

device = 'cuda'
config = default_config

def pre_data(batch_size, num_workers, test_vendor):
    test_vendor = test_vendor

    _, _, _, _, _, _, test_dataset = get_meta_split_data_loaders(
            test_vendor=test_vendor,config=config)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=False)

    print("length of test_dataset", len(test_dataset))

    return test_loader, len(test_dataset)


def save_once(image, pred, mask, model, domain, image_slice):
    pred = pred[:,0,:,:]
    real_mask = mask[:,0,:,:]
    mask = im_convert(real_mask, False)
    image = im_convert(image, False)
    pred = im_convert(pred, False)
    save_dir1 = './pic_scgm/{}/{}/real_mask/'.format(model,domain)
    save_dir2 = './pic_scgm/{}/{}/image/'.format(model,domain)
    save_dir3 = './pic_scgm/{}/{}/pred/'.format(model,domain)
    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)
    os.makedirs(save_dir3, exist_ok=True)
    
    save_image(mask,'{}/{}.png'.format(save_dir1,image_slice))
    save_image(image,'{}/{}.png'.format(save_dir2,image_slice))
    save_image(pred,'{}/{}.png'.format(save_dir3,image_slice))
    
def draw_many_img(model_path_l, test_loader, model, save=False):
    model_l = torch.load(model_path_l, map_location=device)
    # model_r = torch.load(model_path_r, map_location=device)
    model_l = model_l.to(device)
    # model_r = model_r.to(device)
    model_l.eval()
    # model_r.eval()

    flag = '' #!!! '047'
    tot = 0
    tot_sub = []
    files = []
    for minibatch in tqdm(test_loader):
        imgs = minibatch['img']
        mask = minibatch['mask']
        path_img = minibatch['path_img']
        imgs = imgs.to(device)
        mask = mask.to(device)
        domain = path_img[0][-18:-11]
        image_slice = path_img[0][-10:-4]
        with torch.no_grad():
            logits_l,_ = model_l(imgs)
            # logits_r,_ = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        # sof_r = F.softmax(logits_r, dim=1)

        pred = sof_l #!!!(sof_l + sof_r) / 2
        pred = (pred > 0.5).float()

        if save:
            save_once(imgs, pred, mask, model, domain, image_slice)

        # dice score
        tot = dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()

        tot_sub.append(tot)
        files.append(image_slice)
    #!!!print(tot_sub,'popop',len(tot_sub))
    
    with open('./pic_scgm/{}/{}/dice_perimg_{}.txt'.format(model,domain,domain),'a') as f:
        for i in range(len(files)):
            l = '{} {}\n'.format(files[i], tot_sub[i])
            f.writelines(l)

    print('dice is ', sum(tot_sub)/len(tot_sub))

def inference_dual(model_path_l, test_loader):

    model_l = torch.load(model_path_l, map_location=device)
    model_l = model_l.to(device)
    model_l.eval()

    # model_r = torch.load(model_path_r, map_location=device)
    # model_r = model_r.to(device)
    # model_r.eval()

    tot = []
    tot_sub = []
    flag = '000'
    record_flag = {}

    for minibatch in tqdm(test_loader):
        imgs = minibatch['img']
        mask = minibatch['mask']
        path_img = minibatch['path_img']
        imgs = imgs.to(device)
        mask = mask.to(device)
        # print(flag)
        # print(path_img[0][-10: -7])
        if path_img[0][-10: -7] != flag:
            score = sum(tot_sub)/len(tot_sub)
            tot.append(score)
            tot_sub = []

            if score <= 0.7:
                record_flag[flag] = score
            flag = path_img[0][-10: -7]

        with torch.no_grad():
            logits_l, _ = model_l(imgs)
            # logits_r, _ = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        # sof_r = F.softmax(logits_r, dim=1)
        pred = sof_l #!!!(sof_l + sof_r) / 2
        pred = (pred > 0.5).float()
        b, c, h, w = mask.shape
        dice = dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()
        
        tot_sub.append(dice)
    tot.append(sum(tot_sub)/len(tot_sub))

    for i in range(len(tot)):
        tot[i] = tot[i] * 100

    print(tot)
    print(len(tot))
    print(sum(tot)/len(tot))
    print(statistics.stdev(tot))
    print(record_flag)

'''计算的dice有问题，暂时不知道为什么
def cal_dice_perimg(model, domain):
    save_dir1 = './pic_scgm/{}/{}/real_mask/'.format(model,domain)
    save_dir2 = './pic_scgm/{}/{}/pred/'.format(model,domain)
    
    files = os.listdir(save_dir1)
    
    tot_sub = []
    for filename in files:
        path_gt = '{}/{}'.format(save_dir1,filename)
        path_pred = '{}/{}'.format(save_dir2,filename)
        
        path_gt = 'mask.png'
        path_pred = 'pred.png'
        mask = cv2.imread(path_gt)
        pred = cv2.imread(path_pred)
        print(mask.shape,pred.shape)
        mask = torch.from_numpy(mask[:,:,0])
        pred = torch.from_numpy(pred[:,:,0])
        mask[mask==255] = 1
        pred[pred==255] = 1
        print(mask.shape,pred.shape,torch.unique(mask),torch.unique(pred))
        import torchvision
        torchvision.utils.save_image(mask.float(),'mask_af.png')
        torchvision.utils.save_image(pred.float(),'pred_af.png')
        # dice score
        tot = dice_coeff(mask.unsqueeze(0), mask.unsqueeze(0), device).item()
        print(tot,filename, domain)
        print(oo)

        tot_sub.append(tot)
    #!!!print(tot_sub,'popop',len(tot_sub))

    print('dice is ', sum(tot_sub)/len(tot_sub))
    
    with open('./pic_scgm/{}/{}/dice_perimg.txt'.format(model,domain),'a') as f:
        for i in len(files):
            l = ['{} {}'.format(files[i], tot_sub[i])]
            f.writelines(l)'''
    
def main():
    batch_size = 1
    num_workers = 4
    test_vendor = 'A' #!!!'D'

    '''
    model_path_l = './tmodel/l_2%_'+str(test_vendor)+'.pt'
    model_path_r = './tmodel/r_2%_'+str(test_vendor)+'.pt'
    '''
    modeltype = '1gpu/cutmixl2u_instancecpl2u_feamixl2u_deeplabv3_epldiceloss_savebestema_loadselftrainparam_A/nousepseudoloss_lr0.01wdecay0.01_alpha0.4_ema0.9_seed14'
    #!!!baseline_instance_deeplabv3_epldiceloss_savebestema_counttime_B/'
    
    saveimg = False #False #True
    savedirs= './tmodel_scgm/{}'.format(modeltype)
    ratio=0.2
    if 'baseline' in modeltype and 'instance' not in modeltype:
        model_path_l = '{}/SCGM_fixmatch_ratio{}_{}_CM.pt'.format(savedirs,ratio,test_vendor)
    else:
        model_path_l = '{}/stu_SCGM_fixmatch_ratio{}_{}_CM.pt'.format(savedirs,ratio,test_vendor)
    

    test_loader, num_test_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vendor)

    # id = '014123'
    # id = '000003'
    # img_path = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorA/'+ id +'.npz'
    # mask_path = '/home/listu/code/semi_medical/mnms_split_2D/mask/Labeled/vendorA/'+ id +'.png'
    # re_path = '/home/listu/code/semi_medical/mnms_split_2D_re/Labeled/vendorA/'+ id +'.npz'
    # fourier_path = '/home/listu/code/semi_medical/mnms_split_2D/data/Labeled/vendorB/center2/000005.npz'
    # one_image_data = OneImageFolder(img_path, mask_path, re_path, fourier_path)
    # one_image_loader = DataLoader(dataset=one_image_data, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    # draw_img(model_path_l, model_path_r, test_loader, test_vendor)
    if saveimg:
        draw_many_img(model_path_l, test_loader,modeltype,saveimg)
    inference_dual(model_path_l, test_loader)

if __name__ == '__main__':
    main()
