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
from config_scgm_loadaugbaseline_1gpu_A import default_config #!!!from config import default_config
from utils.data_utils import save_image
from utils.dice_loss import dice_coeff
#!!!from draw_dataloader import OneImageFolder

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

    print(tot)
    print(len(tot))
    print(sum(tot)/len(tot))
    print(statistics.stdev(tot))
    print(record_flag)

def main():
    batch_size = 1
    num_workers = 4
    test_vendor = 'A' #!!!'D'

    '''
    model_path_l = './tmodel/l_2%_'+str(test_vendor)+'.pt'
    model_path_r = './tmodel/r_2%_'+str(test_vendor)+'.pt'
    '''
    modeltype = 'default_loadaugbaseline_A/'
    saveimg = False
    savedirs= './tmodel_scgm/{}'.format(modeltype) #default_1gpu_4090' #default_nofft_nostrongaug_1gpu' #!!'tmodel_scgm_4gpu/default' #'./tmodel_scgm/cda_imgcut_l2u0_u02l_feamixup_fixmixlambda0.5_lossmix_lessepoch_B' #_fixmixlambda0.5_lossmix_lessmixlossw_otherw3_mixlossw1_4gpu_4090'
    ratio=0.2
    model_path_l = '{}/l_SCGM_deeplab_ratio{}_{}_CM.pt'.format(savedirs,ratio,test_vendor)
    model_path_r = '{}/r_SCGM_deeplab_ratio{}_{}_CM.pt'.format(savedirs,ratio,test_vendor)
    

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
        draw_many_img(model_path_l, model_path_r, test_loader, modeltype, saveimg)
    inference_dual(model_path_l, model_path_r, test_loader)

if __name__ == '__main__':
    main()
