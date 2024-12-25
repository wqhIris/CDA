import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erfinv
import torch.nn.functional as F
from skimage import measure
from random import choice
import os
from scipy.ndimage import map_coordinates, fourier_gaussian
import albumentations as A
import cv2



def tumor_cp_augmentation(data, label, lung, tumor_dir,img_size=512):
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    
    data, label, lung = data.cpu().numpy(), label.cpu().numpy(), lung.cpu().numpy()
    mixed_data = np.zeros_like(data)
    mixed_label = np.zeros_like(label)
    for i in range(data.shape[0]):
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
        
        if np.sum(lung_i==1)>20:
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.where(lung_i==1)
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break   
            #paste
            copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(3,axis=0)
            # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
            data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
            label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
            lung_i_1 = np.expand_dims(lung_i,0).repeat(3,axis=0)
            data_i[lung_i_1==0]=0
            label_i[lung_i==0]=0
        mixed_data[i,:]=data_i
        mixed_label[i,:]=label_i
    return torch.from_numpy(mixed_data), torch.from_numpy(mixed_label)


def tumor_cp_augmentation_transform(data, label, lung, tumor_dir,img_size=512):
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    
    data, label, lung = data.cpu().numpy(), label.cpu().numpy(), lung.cpu().numpy()
    mixed_data = np.zeros_like(data)
    mixed_label = np.zeros_like(label)
    for i in range(data.shape[0]):
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        # rigid transformation: scaling, rotation, and mirroring
        transform = A.Compose([
                        A.Rotate(limit=90, p=0.5),
                        A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
                        A.Flip(p=0.5)])
        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
        #                 A.Flip(p=0.5),
        #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
        copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
        transformed_image = transform(image=copy_data, mask=copy_tumor)
        copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
        copy_tumor = transformed_image['mask']
        # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
        
        if np.sum(lung_i==1)>20:
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.where(lung_i==1)
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break   
            #paste
            copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(3,axis=0)
            # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
            data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
            label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
            lung_i_1 = np.expand_dims(lung_i,0).repeat(3,axis=0)
            data_i[lung_i_1==0]=0
            label_i[lung_i==0]=0
        mixed_data[i,:]=data_i
        mixed_label[i,:]=label_i
    return torch.from_numpy(mixed_data), torch.from_numpy(mixed_label)        


def tumor_cp_augmentation_transformV1(data, label, lung, tumor_dir,img_size=512):
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    
    data, label, lung = data.cpu().numpy(), label.cpu().numpy(), lung.cpu().numpy()
    mixed_data = np.zeros_like(data)
    mixed_label = np.zeros_like(label)
    for i in range(data.shape[0]):
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        # # rigid transformation: scaling, rotation, and mirroring
        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.Flip(p=0.5)])
        # elastic
        transform = A.Compose([
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)])

        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
        #                 A.Flip(p=0.5),
        #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
        copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
        transformed_image = transform(image=copy_data, mask=copy_tumor)
        copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
        copy_tumor = transformed_image['mask']
        # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
        
        if np.sum(lung_i==1)>20:
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.where(lung_i==1)
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break   
            #paste
            copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(3,axis=0)
            # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
            data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
            label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
            lung_i_1 = np.expand_dims(lung_i,0).repeat(3,axis=0)
            data_i[lung_i_1==0]=0
            label_i[lung_i==0]=0
        mixed_data[i,:]=data_i
        mixed_label[i,:]=label_i
    return torch.from_numpy(mixed_data), torch.from_numpy(mixed_label)

#!!!def tumor_cp_augmentation_transformV2(data, label, lung, tumor_dir,img_size=512):
def tumor_cp_augmentation_transformV2(data, label, lung, tumor_dir, blur_limit=(3,7), img_size=512):
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    
    data, label, lung = data.cpu().numpy(), label.cpu().numpy(), lung.cpu().numpy()
    mixed_data = np.zeros_like(data)
    mixed_label = np.zeros_like(label)
    for i in range(data.shape[0]):
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        # # rigid transformation: scaling, rotation, and mirroring
        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.Flip(p=0.5)])
        # elastic
        # transform = A.Compose([
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)])
        # blur
        #!!!
        if isinstance(blur_limit,list):
            if len(blur_limit) == 1:
                blur_limit = blur_limit[0]
            else:
                blur_limit = tuple(blur_limit)
        transform = A.Compose([
                        A.GaussianBlur(blur_limit=blur_limit,p=0.5),])
                        #!!!A.GaussianBlur(blur_limit=(3,7),p=0.5),])
        #!!!

        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
        #                 A.Flip(p=0.5),
        #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
        copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
        transformed_image = transform(image=copy_data, mask=copy_tumor)
        copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
        copy_tumor = transformed_image['mask']
        # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
        
        if np.sum(lung_i==1)>20:
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.where(lung_i==1)
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break   
            #paste
            '''
            import torchvision
            torchvision.utils.save_image(torch.from_numpy(data_i[0,:,:]).unsqueeze(0).unsqueeze(0), 'aaa5_data{}_all_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(data_i[0,x:(x + cutmix_w),y:(y + cutmix_h)]).unsqueeze(0).unsqueeze(0), 'aaa5_data{}_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(copy_data[0,:,:]).unsqueeze(0).unsqueeze(0), 'aaa6_copydata{}_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(copy_tumor).float().unsqueeze(0).unsqueeze(0), 'aaa7_copytumor{}_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(label_i).float().unsqueeze(0), 'aaa8_label{}_all_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(label_i[x:(x + cutmix_w),y:(y + cutmix_h)]).float().unsqueeze(0), 'aaa8_label{}_img.png'.format(i))
            '''
            
            copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(3,axis=0)
            # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
            copy_data_i = data_i.copy()
            data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
            label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
            copy_data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = copy_data
            '''import torchvision
            torchvision.utils.save_image(torch.from_numpy(copy_data_i),'copy_data_i.png')
            torchvision.utils.save_image(torch.from_numpy(data_i),'data_i.png')
            torchvision.utils.save_image(torch.from_numpy(copy_tumor).float().unsqueeze(0).unsqueeze(0), 'copy_tumor.png')
            aaa = copy_tumor_1*copy_data
            bbb = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)]
            ccc = data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)]
            torchvision.utils.save_image(torch.from_numpy(aaa),'data_i_aaa.png')
            torchvision.utils.save_image(torch.from_numpy(bbb),'data_i_bbb.png')
            torchvision.utils.save_image(torch.from_numpy(ccc),'data_i_ccc.png')
            print(jj)'''
            
            lung_i_1 = np.expand_dims(lung_i,0).repeat(3,axis=0)
            data_i[lung_i_1==0]=0
            label_i[lung_i==0]=0
        mixed_data[i,:]=data_i
        mixed_label[i,:]=label_i
        '''
        import torchvision
        torchvision.utils.save_image(torch.from_numpy(mixed_data), 'aaa3_mix_data_img.png')
        torchvision.utils.save_image(torch.from_numpy(mixed_label).unsqueeze(1).float(), 'aaa4_mix_label_img.png')
        '''
    return torch.from_numpy(mixed_data), torch.from_numpy(mixed_label)
#!!!-------------------------
def tumor_cp_augmentation_transformV2_nopseudo(data, lung, tumor_dir, blur_limit, img_size=512):
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    
    #!!!data, label, lung = data.cpu().numpy(), label.cpu().numpy(), lung.cpu().numpy()
    data, lung = data.cpu().numpy(), lung.cpu().numpy()
    mixed_data = np.zeros_like(data)
    #!!!mixed_label = np.zeros_like(label)
    for i in range(data.shape[0]):
        data_i = data[i,:]
        #!!!label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        # # rigid transformation: scaling, rotation, and mirroring
        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.Flip(p=0.5)])
        # elastic
        # transform = A.Compose([
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)])
        # blur
        #!!!
        if isinstance(blur_limit,list):
            if len(blur_limit) == 1:
                blur_limit = blur_limit[0]
            else:
                blur_limit = tuple(blur_limit)
        transform = A.Compose([
                        A.GaussianBlur(blur_limit=blur_limit,p=0.5),])
                        #!!!A.GaussianBlur(blur_limit=(3,7),p=0.5),])
        #!!!

        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
        #                 A.Flip(p=0.5),
        #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
        copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
        transformed_image = transform(image=copy_data, mask=copy_tumor)
        copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
        copy_tumor = transformed_image['mask']
        # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
        
        if np.sum(lung_i==1)>20:
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.where(lung_i==1)
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break   
            #paste
            '''
            import torchvision
            torchvision.utils.save_image(torch.from_numpy(data_i[0,:,:]).unsqueeze(0).unsqueeze(0), 'aaa5_data{}_all_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(data_i[0,x:(x + cutmix_w),y:(y + cutmix_h)]).unsqueeze(0).unsqueeze(0), 'aaa5_data{}_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(copy_data[0,:,:]).unsqueeze(0).unsqueeze(0), 'aaa6_copydata{}_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(copy_tumor).float().unsqueeze(0).unsqueeze(0), 'aaa7_copytumor{}_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(label_i).float().unsqueeze(0), 'aaa8_label{}_all_img.png'.format(i))
            torchvision.utils.save_image(torch.from_numpy(label_i[x:(x + cutmix_w),y:(y + cutmix_h)]).float().unsqueeze(0), 'aaa8_label{}_img.png'.format(i))
            '''
            
            copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(3,axis=0)
            # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
            copy_data_i = data_i.copy()
            data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
            #!!!label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
            copy_data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = copy_data
            '''import torchvision
            torchvision.utils.save_image(torch.from_numpy(copy_data_i),'copy_data_i.png')
            torchvision.utils.save_image(torch.from_numpy(data_i),'data_i.png')
            torchvision.utils.save_image(torch.from_numpy(copy_tumor).float().unsqueeze(0).unsqueeze(0), 'copy_tumor.png')
            aaa = copy_tumor_1*copy_data
            bbb = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)]
            ccc = data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)]
            torchvision.utils.save_image(torch.from_numpy(aaa),'data_i_aaa.png')
            torchvision.utils.save_image(torch.from_numpy(bbb),'data_i_bbb.png')
            torchvision.utils.save_image(torch.from_numpy(ccc),'data_i_ccc.png')
            print(jj)'''
            
            lung_i_1 = np.expand_dims(lung_i,0).repeat(3,axis=0)
            data_i[lung_i_1==0]=0
            #!!!label_i[lung_i==0]=0
        mixed_data[i,:]=data_i
        #!!!mixed_label[i,:]=label_i
        '''
        import torchvision
        torchvision.utils.save_image(torch.from_numpy(mixed_data), 'aaa3_mix_data_img.png')
        torchvision.utils.save_image(torch.from_numpy(mixed_label).unsqueeze(1).float(), 'aaa4_mix_label_img.png')
        '''
    return torch.from_numpy(mixed_data) #!!!return torch.from_numpy(mixed_data), torch.from_numpy(mixed_label)
#!!!--------------------------

# def tumor_cp_augmentation_transformV2_debug(data, label, lung, tumor_dir,img_size=512):
#     tumor_paths = []
#     for filename in os.listdir(tumor_dir):
#         if '_tumor_' in filename:  
#             tumor_paths.append(os.path.join(tumor_dir,filename))
#     # print('len(tumor_paths):',len(tumor_paths))
#     # print('len(tumor_paths):',len(tumor_paths), tumor_dir,'=--==--==')
#     print(tumor_paths,'==-')
    
    
#     data, label, lung = data.cpu().numpy(), label.cpu().numpy(), lung.cpu().numpy()
#     mixed_data = np.zeros_like(data)
#     mixed_label = np.zeros_like(label)
#     from collections import defaultdict
#     tmp = defaultdict(list)
#     for i in range(data.shape[0]):
#         data_i = data[i,:]
#         label_i = label[i,:]
#         lung_i = lung[i,:]

#         copy_tumor_path = choice(tumor_paths)
#         copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
#         print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

#         copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
#         copy_data = np.load(copy_data_path, allow_pickle=True)
#         # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
#         # # rigid transformation: scaling, rotation, and mirroring
#         # transform = A.Compose([
#         #                 A.Rotate(limit=90, p=0.5),
#         #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
#         #                 A.Flip(p=0.5)])
#         # elastic
#         # transform = A.Compose([
#         #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)])
#         # blur
#         transform = A.Compose([
#                         A.GaussianBlur(blur_limit=(3,7),p=0.5),])

#         # transform = A.Compose([
#         #                 A.Rotate(limit=90, p=0.5),
#         #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
#         #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
#         #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
#         #                 A.Flip(p=0.5),
#         #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
#         copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
#         transformed_image = transform(image=copy_data, mask=copy_tumor)
#         copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
#         copy_tumor = transformed_image['mask']
#         print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
#         cutmix_w = copy_tumor.shape[0]
#         cutmix_h = copy_tumor.shape[1]
        
#         if np.sum(lung_i==1)>20:
#             iter_num = 0
            
#             # print('np.sum(lung_i==1):',np.sum(lung_i==1))
#             mask_voxel_coords = np.where(lung_i==1)
            
#             # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
#             minxidx = int(np.min(mask_voxel_coords[0])) 
#             maxxidx = int(np.max(mask_voxel_coords[0])) 
#             minyidx = int(np.min(mask_voxel_coords[1])) 
#             maxyidx = int(np.max(mask_voxel_coords[1])) 
#             # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
#             mask = torch.zeros(img_size, img_size)
#             # iter_num = 0
#             while True:
#                 iter_num += 1
#                 if iter_num>20:
#                     x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
#                     if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
#                         break
#                 else:
#                     index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
#                     x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
#                     if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
#                         break   
#             #paste
#             print(x,y,cutmix_w,cutmix_h,'[[---')
#             import torchvision
#             torchvision.utils.save_image(torch.from_numpy(data_i[0,:,:]).unsqueeze(0).unsqueeze(0), 'aaa5_data{}_all_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(data_i[0,x:(x + cutmix_w),y:(y + cutmix_h)]).unsqueeze(0).unsqueeze(0), 'aaa5_data{}_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(copy_data[0,:,:]).unsqueeze(0).unsqueeze(0), 'aaa6_copydata{}_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(copy_tumor).float().unsqueeze(0).unsqueeze(0), 'aaa7_copytumor{}_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(label_i).float().unsqueeze(0), 'aaa8_label{}_all_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(label_i[x:(x + cutmix_w),y:(y + cutmix_h)]).float().unsqueeze(0), 'aaa8_label{}_img_debug.png'.format(i))
            
#             copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(3,axis=0)
#             # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
#             data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
#                                                           copy_tumor_1*copy_data
#             label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
#             lung_i_1 = np.expand_dims(lung_i,0).repeat(3,axis=0)
#             torchvision.utils.save_image(torch.from_numpy(data_i[0,:,:]).unsqueeze(0), 'aaa5before_data{}_all_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(label_i).float().unsqueeze(0), 'aaa8before_label{}_img_debug.png'.format(i))
#             data_i[lung_i_1==0]=0
#             label_i[lung_i==0]=0
            
#             torchvision.utils.save_image(torch.from_numpy(data_i[0,:,:]).unsqueeze(0), 'aaa5new_data{}_all_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(label_i).float().unsqueeze(0), 'aaa8new_label{}_img_debug.png'.format(i))
#             torchvision.utils.save_image(torch.from_numpy(lung_i).float().unsqueeze(0), 'aaa8_lung{}_img_debug.png'.format(i))
#         mixed_data[i,:]=data_i
#         mixed_label[i,:]=label_i
#         import torchvision
#         torchvision.utils.save_image(torch.from_numpy(mixed_data), 'aaa3_mix_data_img_debug.png')
#         torchvision.utils.save_image(torch.from_numpy(mixed_label).unsqueeze(1).float(), 'aaa4_mix_label_img_debug.png')
#         '''
#         tmp['x'].append(x)
#         tmp['y'].append(y)
#         tmp['cutmix_w'].append(cutmix_w)
#         tmp['cutmix_h'].append(cutmix_h)
#         tmp['copy_tumor_1'].append(copy_tumor_1)
#         tmp['copy_tumor'].append(copy_tumor)
#         tmp['copy_data'].append(copy_data)
#         '''
    
#     return torch.from_numpy(mixed_data), torch.from_numpy(mixed_label) #,tmp

# def tumor_cp_augmentation_transformV2_fortensor(data, label, lung, tumor_dir,img_size=512,tmp=None):
#     tumor_paths = []
#     for filename in os.listdir(tumor_dir):
#         if '_tumor_' in filename:  
#             tumor_paths.append(os.path.join(tumor_dir,filename))
#     # print('len(tumor_paths):',len(tumor_paths))
    
    
#     mixed_data = torch.zeros(data.size()).to(data.device)
#     mixed_label = torch.zeros(label.size()).to(data.device)
#     for i in range(data.shape[0]):
#         data_i = data[i,:].clone()
#         label_i = label[i,:].clone()
#         lung_i = lung[i,:].clone()

#         copy_tumor_path = choice(tumor_paths)
#         copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
#         # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

#         copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
#         copy_data = np.load(copy_data_path, allow_pickle=True)
#         # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
#         # # rigid transformation: scaling, rotation, and mirroring
#         # transform = A.Compose([
#         #                 A.Rotate(limit=90, p=0.5),
#         #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
#         #                 A.Flip(p=0.5)])
#         # elastic
#         # transform = A.Compose([
#         #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)])
#         # blur
#         transform = A.Compose([
#                         A.GaussianBlur(blur_limit=(3,7),p=0.5),])

#         # transform = A.Compose([
#         #                 A.Rotate(limit=90, p=0.5),
#         #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
#         #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
#         #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
#         #                 A.Flip(p=0.5),
#         #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
#         copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
#         transformed_image = transform(image=copy_data, mask=copy_tumor)
#         copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
#         copy_data = copy_data[0]
#         copy_data = torch.from_numpy(copy_data).to(data_i.device)
#         copy_data = copy_data.unsqueeze(0).repeat(data_i.size(0),1,1)
#         copy_tumor = transformed_image['mask']
#         # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
#         cutmix_w = copy_tumor.shape[0]
#         cutmix_h = copy_tumor.shape[1]
                
        
#         if torch.sum(lung_i==1)>20:
#             iter_num = 0
#             # print('np.sum(lung_i==1):',np.sum(lung_i==1))
#             mask_voxel_coords = np.array(torch.nonzero(lung_i==1).tolist())
#             mask_voxel_coords_x = [x[0] for x in mask_voxel_coords]
#             mask_voxel_coords_y = [x[1] for x in mask_voxel_coords]
#             mask_voxel_coords_x = np.array(mask_voxel_coords_x) 
#             mask_voxel_coords_y = np.array(mask_voxel_coords_y) 
#             mask_voxel_coords = (mask_voxel_coords_x, mask_voxel_coords_y)
            
#             # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
#             minxidx = int(np.min(mask_voxel_coords[0])) 
#             maxxidx = int(np.max(mask_voxel_coords[0])) 
#             minyidx = int(np.min(mask_voxel_coords[1])) 
#             maxyidx = int(np.max(mask_voxel_coords[1])) 
#             # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
#             mask = torch.zeros(img_size, img_size)
#             # iter_num = 0
#             while True:
#                 iter_num += 1
#                 if iter_num>20:
#                     x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
#                     if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
#                         break
#                 else:
#                     index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
#                     x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
#                     if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
#                         break   
#             #paste
#             copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(data_i.size(0),axis=0)
#             # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
#             copy_tumor_1 = torch.from_numpy(copy_tumor_1).to(data_i.device)
#             copy_tumor = torch.from_numpy(copy_tumor).to(data_i.device)
            
#             data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
#                                                           copy_tumor_1*copy_data
#             label_i = label_i.to(data_i.device)
#             label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
#             lung_i_1 = lung_i.repeat(data_i.size(0),1,1)
            
#             data_i[lung_i_1==0]=0
#             label_i[lung_i==0]=0
        
#         mixed_data[i,:]=data_i
#         mixed_label[i,:]=label_i
        
#     return mixed_data, mixed_label



def tumor_cp_augmentation_transformV2_fortensor_v2(data, label, lung, tumor_dir,img_size=512,tmp=None):
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    # print('len(tumor_paths):',len(tumor_paths), tumor_dir,'---[-[-00')
    print(len(tumor_paths),'==-888')
    
    
    mixed_data = torch.zeros(data.size()).to(data.device)
    mixed_label = torch.zeros(label.size()).to(data.device)
    import torchvision
    torchvision.utils.save_image(data[:,0,:,:].unsqueeze(1), 'aaa5_data_all.png')
    for i in range(data.shape[0]):
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        # # rigid transformation: scaling, rotation, and mirroring
        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.Flip(p=0.5)])
        # elastic
        # transform = A.Compose([
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)])
        # blur
        transform = A.Compose([
                        A.GaussianBlur(blur_limit=(3,7),p=0.5),])

        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
        #                 A.Flip(p=0.5),
        #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
        copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
        transformed_image = transform(image=copy_data, mask=copy_tumor)
        copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
        copy_data = copy_data[0]
        copy_data = torch.from_numpy(copy_data).to(data_i.device)
        copy_data = copy_data.unsqueeze(0).repeat(data_i.size(0),1,1)
        copy_tumor = transformed_image['mask']
        print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
                
        
        if torch.sum(lung_i==1)>20:
            iter_num = 0
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.array(torch.nonzero(lung_i==1).tolist())
            mask_voxel_coords_x = [x[0] for x in mask_voxel_coords]
            mask_voxel_coords_y = [x[1] for x in mask_voxel_coords]
            mask_voxel_coords_x = np.array(mask_voxel_coords_x) 
            mask_voxel_coords_y = np.array(mask_voxel_coords_y) 
            mask_voxel_coords = (mask_voxel_coords_x, mask_voxel_coords_y)
            
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            # iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break  
            print(x,y,cutmix_w,cutmix_h)
            
            #paste
            copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(data_i.size(0),axis=0)
            # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
            copy_tumor_1 = torch.from_numpy(copy_tumor_1).to(data_i.device)
            copy_tumor = torch.from_numpy(copy_tumor).to(data_i.device)
            
            
            import torchvision
            torchvision.utils.save_image(data_i[0,:,:].unsqueeze(0).unsqueeze(0), 'aaa5_data{}_all.png'.format(i))
            torchvision.utils.save_image(data_i[0,x:(x + cutmix_w),y:(y + cutmix_h)].unsqueeze(0).unsqueeze(0), 'aaa5_data{}.png'.format(i))
            torchvision.utils.save_image(copy_data[0,:,:].unsqueeze(0).unsqueeze(0), 'aaa6_copydata{}.png'.format(i))
            torchvision.utils.save_image(copy_tumor.float().unsqueeze(0).unsqueeze(0), 'aaa7_copytumor{}.png'.format(i))
            torchvision.utils.save_image(label_i.float().unsqueeze(0), 'aaa8_label{}_all.png'.format(i))
            torchvision.utils.save_image(label_i[x:(x + cutmix_w),y:(y + cutmix_h)].float().unsqueeze(0), 'aaa8_label{}.png'.format(i))
            data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
            label_i = label_i.to(data_i.device)
            label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
            lung_i_1 = lung_i.repeat(data_i.size(0),1,1)
            torchvision.utils.save_image(data_i.unsqueeze(0), 'aaa5before_data{}_all.png'.format(i))
            torchvision.utils.save_image(label_i.float().unsqueeze(0), 'aaa8before_label{}.png'.format(i))
            
            data_i[lung_i_1==0]=0
            label_i[lung_i==0]=0
            
            
            torchvision.utils.save_image(data_i.unsqueeze(0), 'aaa5new_data{}_all.png'.format(i))
            torchvision.utils.save_image(label_i.float().unsqueeze(0), 'aaa8new_label{}.png'.format(i))
            torchvision.utils.save_image(lung_i.float().unsqueeze(0), 'aaa8_lung{}.png'.format(i))
            
        '''
        x = tmp['x'][i]
        y = tmp['y'][i]
        cutmix_w = tmp['cutmix_w'][i]
        cutmix_h = tmp['cutmix_h'][i]
        copy_tumor_1 = tmp['copy_tumor_1'][i]
        copy_tumor = tmp['copy_tumor'][i]
        copy_data = tmp['copy_data'][i]
        copy_data = torch.from_numpy(copy_data).to(data_i.device)
        copy_tumor_1 = torch.from_numpy(copy_tumor_1).to(data_i.device)
        copy_tumor = torch.from_numpy(copy_tumor).to(data_i.device)
        label_i = label_i.to(data_i.device)    
        
        import torchvision
        torchvision.utils.save_image(data_i[0,:,:].unsqueeze(0).unsqueeze(0), 'aaa5_data{}_all.png'.format(i))
        torchvision.utils.save_image(data_i[0,x:(x + cutmix_w),y:(y + cutmix_h)].unsqueeze(0).unsqueeze(0), 'aaa5_data{}.png'.format(i))
        torchvision.utils.save_image(copy_data[0,:,:].unsqueeze(0).unsqueeze(0), 'aaa6_copydata{}.png'.format(i))
        torchvision.utils.save_image(copy_tumor.float().unsqueeze(0).unsqueeze(0), 'aaa7_copytumor{}.png'.format(i))
        torchvision.utils.save_image(label_i.float().unsqueeze(0), 'aaa8_label{}_all.png'.format(i))
        torchvision.utils.save_image(label_i[x:(x + cutmix_w),y:(y + cutmix_h)].float().unsqueeze(0), 'aaa8_label{}.png'.format(i))
        data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
        label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
        lung_i_1 = lung_i.repeat(data_i.size(0),1,1)
        data_i[lung_i_1==0]=0
        label_i[lung_i==0]=0
        '''

        
        mixed_data[i,:]=data_i.clone()
        mixed_label[i,:]=label_i.clone()
        torchvision.utils.save_image(mixed_data[:,0,:,:].unsqueeze(1), 'aaa3_mix_data.png')
        torchvision.utils.save_image(mixed_label[:,:,:].unsqueeze(1), 'aaa4_mix_label.png')
        
    return mixed_data, mixed_label



def tumor_cp_augmentation_transformV2_fortensor_loadtumor(data, label, lung, tumor, img_size=512): 
    mixed_data = torch.zeros(data.size()).to(data.device)
    mixed_label = torch.zeros(label.size()).to(data.device)
    for i in range(data.shape[0]):        
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]
        copy_data = tumor[0][i]  #!!![3,h,w]
        copy_tumor = tumor[1][i]   #!!![h,w]   
        # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
                
        
        if torch.sum(lung_i==1)>20:
            iter_num = 0
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.array(torch.nonzero(lung_i==1).tolist())
            mask_voxel_coords_x = [x[0] for x in mask_voxel_coords]
            mask_voxel_coords_y = [x[1] for x in mask_voxel_coords]
            mask_voxel_coords_x = np.array(mask_voxel_coords_x) 
            mask_voxel_coords_y = np.array(mask_voxel_coords_y) 
            mask_voxel_coords = (mask_voxel_coords_x, mask_voxel_coords_y)
            
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            # iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break  
            print(x,y,cutmix_w,cutmix_h)
            #paste
            copy_tumor_1 = np.expand_dims(copy_tumor,0).repeat(data_i.size(0),axis=0) #!!![c,h,w]
            # print('data_i.shape, copy_data.shape, copy_tumor_1.shape:',data_i.shape, copy_data.shape, copy_tumor_1.shape)
            copy_tumor_1 = torch.from_numpy(copy_tumor_1).to(data_i.device)
            copy_tumor = torch.from_numpy(copy_tumor).to(data_i.device)
        
            data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor_1)*data_i[:,x:(x + cutmix_w),y:(y + cutmix_h)] + \
                                                          copy_tumor_1*copy_data
            label_i = label_i.to(data_i.device)
            label_i[x:(x + cutmix_w),y:(y + cutmix_h)] = (1-copy_tumor)*label_i[x:(x + cutmix_w),y:(y + cutmix_h)] + copy_tumor
            lung_i_1 = lung_i.repeat(data_i.size(0),1,1)
            
            data_i[lung_i_1==0]=0
            label_i[lung_i==0]=0
        
        mixed_data[i,:]=data_i.clone()
        mixed_label[i,:]=label_i.clone()
        # import torchvision
        # torchvision.utils.save_image(mixed_data[:,0,:,:].unsqueeze(1), 'aaa3_mix_data.png')
        # torchvision.utils.save_image(mixed_label[:,:,:].unsqueeze(1), 'aaa4_mix_label.png')
        
    return mixed_data, mixed_label





#!!!def tumor_cp_augmentation_transformV2_fortensor_loadcutmixbox(data, label, lung, tumor_dir,img_size=512,tmp=None):
def tumor_cp_augmentation_transformV2_fortensor_loadcutmixbox(data, label, lung, tumor_dir, blur_limit, img_size=512,tmp=None): 
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    # print('len(tumor_paths):',len(tumor_paths), tumor_dir,'---[-[-00')
    # print(len(tumor_paths),'==-888')

    tumor_data_l = []
    tumor_mask_l = []
    x_list = []
    y_list = []
    cutmix_w_list = []
    cutmix_h_list = []
    '''
    import torchvision
    torchvision.utils.save_image(data[:,0,:,:].unsqueeze(1), 'aaa5_data_all.png')
    '''
    for i in range(data.shape[0]):
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        
        # # rigid transformation: scaling, rotation, and mirroring
        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.Flip(p=0.5)])
        # elastic
        # transform = A.Compose([
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)])
        # blur
        #!!!
        if isinstance(blur_limit,list):
            if len(blur_limit) == 1:
                blur_limit = blur_limit[0]
            else:
                blur_limit = tuple(blur_limit)
        transform = A.Compose([
                        A.GaussianBlur(blur_limit=blur_limit,p=0.5),])
                        #!!!A.GaussianBlur(blur_limit=(3,7),p=0.5),])
        #!!!

        # transform = A.Compose([
        #                 A.Rotate(limit=90, p=0.5),
        #                 A.RandomScale(scale_limit=0.25, interpolation=1, always_apply=False, p=0.5),
        #                 A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #                 A.GaussianBlur(blur_limit=(3,7),p=0.5),
        #                 A.Flip(p=0.5),
        #                 A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)])
        copy_data = copy_data.transpose(1,2,0) #(3,512,512)-->(512,512,3)
        transformed_image = transform(image=copy_data, mask=copy_tumor)
        copy_data = transformed_image['image'].transpose(2, 0, 1) #(3,512,512)
        copy_data = copy_data[0]
        copy_data = torch.from_numpy(copy_data).to(data_i.device)
        copy_data = copy_data.unsqueeze(0).repeat(data_i.size(0),1,1)
        copy_tumor = transformed_image['mask']
        # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        tumor_data_l.append(copy_data)
        tumor_mask_l.append(copy_tumor)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
                
        
        if torch.sum(lung_i==1)>20:
            iter_num = 0
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.array(torch.nonzero(lung_i==1).tolist())
            mask_voxel_coords_x = [x[0] for x in mask_voxel_coords]
            mask_voxel_coords_y = [x[1] for x in mask_voxel_coords]
            mask_voxel_coords_x = np.array(mask_voxel_coords_x) 
            mask_voxel_coords_y = np.array(mask_voxel_coords_y) 
            mask_voxel_coords = (mask_voxel_coords_x, mask_voxel_coords_y)
            
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            # iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break  
            # print(x,y,cutmix_w,cutmix_h)
            x_list.append(x)
            y_list.append(y)
            cutmix_w_list.append(cutmix_w)
            cutmix_h_list.append(cutmix_h)
        else:
            x_list.append(None)
            y_list.append(None)
            cutmix_w_list.append(None)
            cutmix_h_list.append(None)
    return x_list, y_list, cutmix_w_list, cutmix_h_list, tumor_data_l, tumor_mask_l




def tumor_cp_augmentation_fortensor_loadcutmixbox(data, label, lung, tumor_dir,img_size=512):
    tumor_paths = []
    for filename in os.listdir(tumor_dir):
        if '_tumor_' in filename:  
            tumor_paths.append(os.path.join(tumor_dir,filename))
    # print('len(tumor_paths):',len(tumor_paths))
    # print('len(tumor_paths):',len(tumor_paths), tumor_dir,'---[-[-00')
    # print(len(tumor_paths),'==-888')

    tumor_data_l = []
    tumor_mask_l = []
    x_list = []
    y_list = []
    cutmix_w_list = []
    cutmix_h_list = []
    '''
    import torchvision
    torchvision.utils.save_image(data[:,0,:,:].unsqueeze(1), 'aaa5_data_all.png')
    '''
    for i in range(data.shape[0]):
        data_i = data[i,:]
        label_i = label[i,:]
        lung_i = lung[i,:]

        copy_tumor_path = choice(tumor_paths)
        copy_data_path = copy_tumor_path.replace('_tumor_','_data_') 
        # print('copy_tumor_path, copy_data_path:', copy_tumor_path, copy_data_path)

        copy_tumor = np.load(copy_tumor_path, allow_pickle=True)
        copy_data = np.load(copy_data_path, allow_pickle=True)
        # print('copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)        
        
        # print('transformed---copy_tumor.shape, copy_data.shape:',copy_tumor.shape, copy_data.shape)
        copy_data = torch.from_numpy(copy_data).to(data_i.device)
        tumor_data_l.append(copy_data)
        tumor_mask_l.append(copy_tumor)
        
        cutmix_w = copy_tumor.shape[0]
        cutmix_h = copy_tumor.shape[1]
                
        
        if torch.sum(lung_i==1)>20:
            iter_num = 0
            # print('np.sum(lung_i==1):',np.sum(lung_i==1))
            mask_voxel_coords = np.array(torch.nonzero(lung_i==1).tolist())
            mask_voxel_coords_x = [x[0] for x in mask_voxel_coords]
            mask_voxel_coords_y = [x[1] for x in mask_voxel_coords]
            mask_voxel_coords_x = np.array(mask_voxel_coords_x) 
            mask_voxel_coords_y = np.array(mask_voxel_coords_y) 
            mask_voxel_coords = (mask_voxel_coords_x, mask_voxel_coords_y)
            
            # print('np.array(mask_voxel_coords[0]).shape:',np.array(mask_voxel_coords[0]).shape)
            minxidx = int(np.min(mask_voxel_coords[0])) 
            maxxidx = int(np.max(mask_voxel_coords[0])) 
            minyidx = int(np.min(mask_voxel_coords[1])) 
            maxyidx = int(np.max(mask_voxel_coords[1])) 
            # print('maxxidx,maxyidx:',maxxidx,maxyidx)
            
            mask = torch.zeros(img_size, img_size)
            # iter_num = 0
            while True:
                iter_num += 1
                if iter_num>20:
                    x, y = np.random.randint(0, img_size), np.random.randint(0, img_size)
                    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                        break
                else:
                    index = choice(range(np.array(mask_voxel_coords[0]).shape[0]))
                    x, y = mask_voxel_coords[0][index], mask_voxel_coords[1][index] #start point
                    if x + cutmix_w <= maxxidx and y + cutmix_h <= maxyidx:
                        break  
            # print(x,y,cutmix_w,cutmix_h)
            x_list.append(x)
            y_list.append(y)
            cutmix_w_list.append(cutmix_w)
            cutmix_h_list.append(cutmix_h)
        else:
            x_list.append(None)
            y_list.append(None)
            cutmix_w_list.append(None)
            cutmix_h_list.append(None)
    return x_list, y_list, cutmix_w_list, cutmix_h_list, tumor_data_l, tumor_mask_l