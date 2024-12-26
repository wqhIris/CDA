import os
#!!!os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import math
import statistics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from networks.scgm_network import my_net
from utils.utils import get_device, check_accuracy, dice_loss, im_convert, label_to_onehot
from scgm_dataloader import get_meta_split_data_loaders
#!!!from config import default_config
from utils.data_utils import save_image
from utils.dice_loss import dice_coeff
#!!!from draw_dataloader import OneImageFolder

device = 'cuda'

def pre_data(batch_size, num_workers, test_vendor):
    test_vendor = test_vendor

    _, _, _, _, _, _, test_dataset = get_meta_split_data_loaders(
            test_vendor=test_vendor)
    
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
    
def draw_many_img(model_path_l, model_path_r, test_loader, model, save=False):
    model_l = torch.load(model_path_l, map_location=device)
    model_r = torch.load(model_path_r, map_location=device)
    model_l = model_l.to(device)
    model_r = model_r.to(device)
    model_l.eval()
    model_r.eval()

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
            logits_r,_ = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)

        pred = (sof_l + sof_r) / 2
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

def inference_dual(model_path_l, model_path_r, test_loader):

    model_l = torch.load(model_path_l, map_location=device)
    model_l = model_l.to(device)
    model_l.eval()

    model_r = torch.load(model_path_r, map_location=device)
    model_r = model_r.to(device)
    model_r.eval()

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
            logits_r, _ = model_r(imgs)

        sof_l = F.softmax(logits_l, dim=1)
        sof_r = F.softmax(logits_r, dim=1)
        pred = (sof_l + sof_r) / 2
        pred = (pred > 0.5).float()
        dice = dice_coeff(pred[:, 0, :, :], mask[:, 0, :, :], device).item()
        tot_sub.append(dice)
    tot.append(sum(tot_sub)/len(tot_sub))

    for i in range(len(tot)):
        tot[i] = tot[i] * 100

    '''print(tot)
    print(len(tot))
    print(sum(tot)/len(tot))
    print(statistics.stdev(tot))
    print(record_flag)'''
    return tot

def main():
    batch_size = 1
    num_workers = 4

    '''
    model_path_l = './tmodel/l_2%_'+str(test_vendor)+'.pt'
    model_path_r = './tmodel/r_2%_'+str(test_vendor)+'.pt'
    '''
    saveimg = False
    savedirs= './tmodel_scgm/{}'
    ratio=0.2   
    
    modeltype = 'default_loadaugbaseline_{}/lr0.0001'
    model_path_l = '{}/l_SCGM_deeplab_ratio{}_{}_CM.pt'
    model_path_r = '{}/r_SCGM_deeplab_ratio{}_{}_CM.pt'
    
    #!!!A
    test_vender_A = 'A'
    modeltype_A = modeltype.format(test_vender_A)
    savedirs_A = savedirs.format(modeltype_A)
    model_path_l_A = model_path_l.format(savedirs_A,ratio,test_vender_A)
    model_path_r_A = model_path_r.format(savedirs_A,ratio,test_vender_A)
    test_loader_A, num_test_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vender_A)
    dice_A = inference_dual(model_path_l_A, model_path_r_A, test_loader_A)
    print('====A:', sum(dice_A)/len(dice_A))
    print('====A:', statistics.stdev(dice_A))
    
    #!!!B
    test_vender_B = 'B'
    modeltype_B = 'default_loadaugbaseline/lr0.0001'
    savedirs_B = savedirs.format(modeltype_B)
    model_path_l_B = model_path_l.format(savedirs_B,ratio,test_vender_B)
    model_path_r_B = model_path_r.format(savedirs_B,ratio,test_vender_B)
    test_loader_B, num_test_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vender_B)
    dice_B = inference_dual(model_path_l_B, model_path_r_B, test_loader_B)
    print('====A:', sum(dice_B)/len(dice_B))
    print('====A:', statistics.stdev(dice_B))
    
    #!!!C
    test_vender_C = 'C'
    modeltype_C = modeltype.format(test_vender_C)
    savedirs_C = savedirs.format(modeltype_C)
    model_path_l_C = model_path_l.format(savedirs_C,ratio,test_vender_C)
    model_path_r_C = model_path_r.format(savedirs_C,ratio,test_vender_C)
    test_loader_C, num_test_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vender_C)
    dice_C = inference_dual(model_path_l_C, model_path_r_C, test_loader_C)
    print('====A:', sum(dice_C)/len(dice_C))
    print('====A:', statistics.stdev(dice_C))
    
    #!!!D
    test_vender_D = 'D'
    modeltype_D = modeltype.format(test_vender_D)
    savedirs_D = savedirs.format(modeltype_D)
    model_path_l_D = model_path_l.format(savedirs_D,ratio,test_vender_D)
    model_path_r_D = model_path_r.format(savedirs_D,ratio,test_vender_D)
    test_loader_D, num_test_imgs = pre_data(
        batch_size=batch_size, num_workers=num_workers, test_vendor=test_vender_D)
    dice_D = inference_dual(model_path_l_D, model_path_r_D, test_loader_D)
    print('====A:', sum(dice_D)/len(dice_D))
    print('====A:', statistics.stdev(dice_D))
    
    dice = dice_A + dice_B + dice_C + dice_D
    print(len(dice))
    print('====:', sum(dice)/len(dice))
    print('====:', statistics.stdev(dice))

if __name__ == '__main__':
    main()
