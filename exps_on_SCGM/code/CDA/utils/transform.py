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

def obtain_cutmix_box(img_size, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.ones(img_size, img_size)
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 0

    return mask

#carvemix
def get_distance(f,spacing=None):
    """Return the signed distance."""

    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, -(dist_func(f,sampling=spacing)),
                        dist_func(1-f,sampling=spacing))

    return distance

def obtain_carve_mask(img_size, label):
    mask = np.zeros((img_size, img_size))
    label = (label==1).cpu().numpy().astype('float32')
    dis_array = get_distance(label)    #creat signed distance
    c = np.random.beta(1, 1)
    c = (c-0.5)*2  # [-1.1] 
    if c>0:
        lam=c*np.min(dis_array)/2              # λl = -1/2|min(dis_array)|
    else:
        lam=c*np.min(dis_array)
    mask = (dis_array<lam).astype('float32')   #creat M
    return torch.from_numpy(mask)

'''def mix(mask, data_l, data_ul):
    # print('mask.shape, data_l.shape, data_ul.shape:',mask.shape, data_l.shape, data_ul.shape)
    # get the random mixing objects
    rand_index = torch.randperm(data_ul.shape[0])[:data_ul.shape[0]]
    #Mix
    data = torch.cat([(mask[i] * data_ul[rand_index[i]] + (1 - mask[i]) * data_l[i]).unsqueeze(0) for i in range(data_l.shape[0])])
    return data'''
def mix(mask, data_l, data_ul, rand_index=None):
    # print('mask.shape, data_l.shape, data_ul.shape:',mask.shape, data_l.shape, data_ul.shape)
    # get the random mixing objects
    if rand_index is None:
        rand_index = torch.randperm(data_ul.shape[0])[:data_ul.shape[0]]
    #Mix
    data = torch.cat([(mask[i] * data_ul[rand_index[i]] + (1 - mask[i]) * data_l[i]).unsqueeze(0) for i in range(data_l.shape[0])])
    return data,rand_index


def mix_reverse(mask, data_l, data_ul):
    # print('mask.shape, data_l.shape, data_ul.shape:',mask.shape, data_l.shape, data_ul.shape)
    # get the random mixing objects
    rand_index = torch.randperm(data_ul.shape[0])[:data_ul.shape[0]]
    #Mix
    data = torch.cat([((1-mask[i]) * data_ul[rand_index[i]] + mask[i] * data_l[i]).unsqueeze(0) for i in range(data_l.shape[0])])
    return data

'''
def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, base_size, ratio_range):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        #sigma = np.random.uniform(0.1, 0.5)
        sigma = 0.1
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=-1, value_max=1, pixel_level=True):
    if random.random() < p:
        img = np.array(img)#.transpose(2,1,0)
        mask = np.array(mask)#.transpose(2,1,0)

        img_h, img_w = img.shape#, img_c

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w))#, img_c
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

        img = Image.fromarray(img.astype(np.float32))#.transpose(2,1,0)
        mask = Image.fromarray(mask.astype(np.uint8))#.transpose(2,1,0)

    return img, mask




def obtain_partialfg_mask_cutmix(img_size, label, task_id, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = np.zeros((img_size, img_size))
    # if random.random() > p:
    #     return mask
    mask = (label==int(task_id)).astype('float32')
    arr = np.nonzero(mask)
    # print('arr:',arr)
    if len(arr[0])==0:
        return torch.from_numpy(mask)
    else:
        minA = arr[0].min()
        maxA = arr[0].max()
        minB = arr[1].min()
        maxB = arr[1].max()
        # size = np.random.uniform(size_min, size_max) * img_size * img_size
        # while True:
        #     ratio = np.random.uniform(ratio_1, ratio_2)
        #     cutmix_w = int(np.sqrt(size / ratio))
        #     cutmix_h = int(np.sqrt(size * ratio))
        #     x = np.random.randint(minA, maxA)
        #     y = np.random.randint(minB, maxB)
        #     if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
        #         break
        # mask[y:y + cutmix_h, x:x + cutmix_w] = 1

        bbox = [int(max(minA - 5, 0)), int(min(maxA + 5 + 1, 256)), int(max(minB - 5, 0)), int(min(maxB + 5 + 1,256))]
        mask[bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
        return torch.from_numpy(mask)




# def obtain_class_mask(img_size, label, task_id, p=0.5):
#     mask = np.zeros((img_size, img_size))
#     if random.random() > p:
#         return mask
#     mask = (label==int(task_id)).astype('float32')
#     return mask

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N




def generate_cow_mask(img_size, sigma, p, seed=None):
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    # Compute threshold
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    return (Ns > t).astype(float) # Apply threshold and return



def mix_batchrandom(mask, data, target, partial_target):
    # get the random mixing objects
    rand_index = torch.randperm(data.shape[0])[:data.shape[0]]
    #Mix
    data = torch.cat([(mask[i] * data[rand_index[i]] + (1 - mask[i]) * data[i]).unsqueeze(0) for i in range(data.shape[0])])
    
    target = torch.cat([(mask[i] * target[rand_index[i]] + (1 - mask[i]) * target[i]).unsqueeze(0) for i in range(target.shape[0])])

    partial_target = torch.cat([(mask[i] * partial_target[rand_index[i]] + (1 - mask[i]) * partial_target[i]).unsqueeze(0) for i in range(partial_target.shape[0])])

    return data, target, partial_target 

 

def mix_reverse(mask, data = None, target = None, partial_target=None):
    #Mix
    if not (data is None):
        data = torch.cat([((1-mask[(i+1) % data.shape[0]])* data[i] + mask[(i+1) % data.shape[0]] * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        
    if not (target is None):
        target = torch.cat([((1-mask[(i+1) % target.shape[0]])* target[i] + mask[(i+1) % target.shape[0]] * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
        
    if not (partial_target is None):
        partial_target = torch.cat([((1-mask[(i+1) % partial_target.shape[0]])* partial_target[i] + mask[(i+1) % partial_target.shape[0]] * partial_target[(i + 1) % partial_target.shape[0]]).unsqueeze(0) for i in range(partial_target.shape[0])])

    return data, target, partial_target 

def copy_paste(mask, data = None, target = None):
    assert data is not None and target is not None
    img_size = data.shape[-1] #256
    for i in range(data.shape[0]):
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        center = [x,y]
        src_data = data[i].squeeze(0)
        tgt_data = data[(i + 1) % data.shape[0]].squeeze(0) #(256,256)
        src_seg = target[i] # single objects
        tgt_seg = target[(i + 1) % target.shape[0]]
        if torch.sum(mask[i])>0:
            patch_length_x, patch_length_y, bboxes = get_bbox_from_mask(mask[i].data.cpu().numpy())
            
            
            # print('-------------------------------------',i)
            # print('tgt_data.shape:',tgt_data.shape)
            start_x, end_x, (ps_x, pe_x) = get_valid_center(center[0], patch_length_x, tgt_data.shape[0])
            start_y, end_y, (ps_y, pe_y) = get_valid_center(center[1], patch_length_y, tgt_data.shape[1])
            seg_patch = src_seg[bboxes[0][0]+ps_x: bboxes[0][1]+pe_x, bboxes[1][0]+ps_y: bboxes[1][1]+pe_y]
            data_patch = src_data[bboxes[0][0]+ps_x: bboxes[0][1]+pe_x, bboxes[1][0]+ps_y: bboxes[1][1]+pe_y]
            mask_patch = mask[i][bboxes[0][0]+ps_x: bboxes[0][1]+pe_x, bboxes[1][0]+ps_y: bboxes[1][1]+pe_y]
            
            tgt_data[start_x:end_x, start_y:end_y]\
                    = (1-mask_patch) * tgt_data[start_x:end_x, start_y:end_y] \
                    + mask_patch * data_patch

            tgt_seg[start_x:end_x, start_y:end_y]\
                    = (1-mask_patch) * tgt_seg[start_x:end_x, start_y:end_y] \
                    + mask_patch * seg_patch
        if i==0:
            cp_data = tgt_data.unsqueeze(0)
            cp_seg = tgt_seg.unsqueeze(0)
        else:
            cp_data = torch.cat([cp_data, tgt_data.unsqueeze(0)],0)
            cp_seg = torch.cat([cp_seg, tgt_seg.unsqueeze(0)],0)
    return cp_data.unsqueeze(1).cuda(), cp_seg.cuda()
        

def get_valid_center(center, patch_length, volume_length):
    patch_offset = [0, 0]
    
    if (center - patch_length//2) > 0:
        start = (center - patch_length//2)
    else:
        start = 0
        patch_offset[0] = patch_length//2 - center
    
    if (center + patch_length - patch_length//2) <= volume_length:
        end = (center + patch_length - patch_length//2)
    else:
        end = volume_length
        patch_offset[1] = -(center + patch_length - patch_length//2 - volume_length)
        
    return start, end, patch_offset          


def get_bbox_from_mask(mask, outside_value=0, pad=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minxidx = int(np.min(mask_voxel_coords[0])) - pad
    maxxidx = int(np.max(mask_voxel_coords[0])) + 1 + pad
    minyidx = int(np.min(mask_voxel_coords[1])) - pad
    maxyidx = int(np.max(mask_voxel_coords[1])) + 1 + pad

    if (maxxidx - minxidx)%2 != 0:
        maxxidx += 1
    if (maxyidx - minyidx)%2 != 0:
        maxyidx += 1
    maxxidx = min(mask.shape[0],maxxidx)
    maxyidx = min(mask.shape[1],maxyidx)
    patch_length_x = maxxidx - minxidx
    patch_length_y = maxyidx - minyidx
    return patch_length_x, patch_length_y, np.array([[minxidx, maxxidx], [minyidx, maxyidx]], dtype=np.int)

'''

