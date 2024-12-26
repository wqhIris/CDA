import numpy as np
import SimpleITK as sitk
import os
import time
from multiprocessing import Pool
from skimage.morphology import label
import random
from scipy import ndimage
import argparse
from skimage.transform import resize
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import pandas as pd
from PIL import Image

TUMOR_LABEL = 1

def get_bbox_from_mask(mask, outside_value=0, pad=0):
    t = time.time()
    mask_voxel_coords = np.where(mask != outside_value)
    minxidx = int(np.min(mask_voxel_coords[0])) - pad
    maxxidx = int(np.max(mask_voxel_coords[0])) + 1 + pad
    minyidx = int(np.min(mask_voxel_coords[1])) - pad
    maxyidx = int(np.max(mask_voxel_coords[1])) + 1 + pad

    if (maxxidx - minxidx)%2 != 0:
        maxxidx += 1
    if (maxyidx - minyidx)%2 != 0:
        maxyidx += 1
    return np.array([[minxidx, maxxidx], [minyidx, maxyidx]], dtype=int)#!!!np.int)

def crop_seg_to_bbox(image, bbox):
    resizer = (slice(bbox[0][0], bbox[0][1]+1), slice(bbox[1][0], bbox[1][1]+1))
    return image[resizer]

def crop_image_to_bbox(image, bbox):
    resizer = (slice(0,3), slice(bbox[0][0], bbox[0][1]+1), slice(bbox[1][0], bbox[1][1]+1))
    return image[resizer]

class TumorExtractor(object):
    def __init__(self, data_root, file_path, save_dir, num_processes=6, dataset_name='COVID249'):
                
        self.num_processes = num_processes
        self.root_path = data_root
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        
        self.file_path = file_path
        excelData = pd.read_excel(self.file_path)
        length = excelData.shape[0] 
        self.paths = []
        for i in range(length): 
            file_name_i =  excelData.iloc[i][0]
            self.paths.append(file_name_i)
        
    def extract(self, case_path):

        case = case_path

        case_img_path = self.root_path + "data/{}/PNG/images/{}".format(self.dataset_name,  case) 
        case_label_path = self.root_path + "data/{}/PNG/labels/{}".format(self.dataset_name, case) 
        case_lung_path = self.root_path + "data/{}/PNG/lung/{}".format(self.dataset_name, case) 

        image = Image.open(case_img_path) #(512,512,3)
        seg = Image.open(case_label_path)
        lung = Image.open(case_lung_path)

        image = (np.asarray(image).astype(np.float32).transpose(2, 0, 1))/255.0 #(3,512,512)
        seg = np.asarray(seg).astype(np.uint8)
        lung = np.asarray(lung).astype(np.uint8)

        # print('image.shape, seg.shape:',image.shape, seg.shape)


        labelmap, numregions = label(seg == TUMOR_LABEL, return_num=True)
        if numregions == 0:
            return

        
        for l in range(1, numregions + 1):
            bbox = get_bbox_from_mask(labelmap==l, pad = 0)
            cropped_data = crop_image_to_bbox(image, bbox)
            cropped_seg = crop_seg_to_bbox(labelmap==l,bbox).astype(np.uint8)
            cropped_seg[cropped_seg>0] = TUMOR_LABEL
            # print('cropped_data.shape, cropped_seg.shape:',cropped_data.shape, cropped_seg.shape)
            out_name = case_path.split('.')
            out_name[0] += '_tumor_'+str(l)
            out_name[1] = 'npy'
            out_name = '.'.join(out_name)
            out_name = os.path.join(self.save_dir,out_name)
            print(out_name)
            np.save(out_name.replace('_tumor_','_data_'), cropped_data)
            np.save(out_name, cropped_seg)

        #### organ position ####
        # we need to find out where the classes are and sample some random locations
        # let's do 10,000 samples per class
        # seed this for reproducibility!
        '''
        num_samples = 10000
        min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(12345)
        all_organ_locs = np.argwhere(seg == ORGAN_LABEL)
        target_num_samples = min(num_samples, len(all_organ_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_organ_locs) * min_percent_coverage)))
        selected = all_organ_locs[rndst.choice(len(all_organ_locs), target_num_samples, replace=False)]

        tumors.append(selected)
        '''

        

    def extract_all(self):
        p = Pool(self.num_processes)
        res = p.map(self.extract, self.paths)
        p.close()
        p.join()


def update_tumor_pool(pseudo_tumor_dir, case_path, image, seg):
    labelmap, numregions = label(seg == TUMOR_LABEL, return_num=True)
    if numregions == 0:
        return

    for l in range(1, numregions + 1):
        bbox = get_bbox_from_mask(labelmap==l, pad = 0)
        cropped_data = crop_image_to_bbox(image, bbox)
        cropped_seg = crop_seg_to_bbox(labelmap==l,bbox).astype(np.uint8)
        cropped_seg[cropped_seg>0] = TUMOR_LABEL
        # print('cropped_data.shape, cropped_seg.shape:',cropped_data.shape, cropped_seg.shape)
        # print('case_path:',case_path)
        out_name = case_path.split('.')
        out_name[0] += '_tumor_'+str(l)
        out_name[1] = 'npy'
        out_name = '.'.join(out_name)
        out_name = os.path.join(pseudo_tumor_dir,out_name)
        np.save(out_name.replace('_tumor_','_data_'), cropped_data)
        np.save(out_name, cropped_seg)

if __name__ == "__main__":

    data_root = '/root/autodl-fs/CDA/exps_on_COVID-19-20/data/'  #!!!'/home/xjl/lx/xx_covid19/DA_covid/'
    file_path = '/root/autodl-fs/CDA/exps_on_COVID-19-20/data/COVID249/train_0.3_l.xlsx' #!!!"/home/xjl/lx/xx_covid19/DA_covid/data/COVID249/train_0.1_l.xlsx"
    save_dir = '/root/autodl-fs/CDA/exps_on_COVID-19-20/data/COVID249/tumor_0.3/' #!!!"/home/xjl/lx/xx_covid19/DA_covid/data/COVID249/tumors_0.1/"
    #!!!
    os.makedirs(save_dir,  exist_ok=True)
    #!!!
    tumor_extractor = TumorExtractor(data_root, file_path, save_dir, num_processes=1)
    tumor_extractor.extract_all()


