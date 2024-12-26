import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

def get_bbox_from_mask(mask, outside_value=0, pad=0):
    mask_voxel_coords = np.where(mask.cpu().numpy() != outside_value)
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

def adjustBright(img, mask):
    img = F.adjust_contrast(img, 2.0)
    return img, mask
'''    
def randomBright(img, mask):
    trans_bright = transforms.ColorJitter(contrast=0.3) #!!!(brightness=0.6,contrast=0.6)
    img = trans_bright(img)    
    return img, mask
'''

def randomRotate(img, mask):    
    trans_rotate = transforms.RandomRotation(45)
    img = trans_rotate(img)
    mask = trans_rotate(mask.unsqueeze(0)).squeeze()
    
    return img, mask

'''
def randomResizeBrightPad(img, mask):
    c, w, h = img.size()
    ratio_range = [1.0, 1.5]
    
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))
    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    trans_im = transforms.Resize((ow,oh), transforms.InterpolationMode.BILINEAR)
    trans_mask = transforms.Resize((ow,oh), transforms.InterpolationMode.NEAREST)

    img = trans_im(img)
    mask = trans_mask(mask.unsqueeze(0)).squeeze()
    
    trans_bright = transforms.ColorJitter(contrast=0.3) #!!!brightness=0.6,contrast=0.6)
    img = trans_bright(img)
    
    _, neww, newh = img.size()
    if neww > w:
        trans_center = transforms.CenterCrop((w, h))
        img = trans_center(img)
        mask = trans_center(mask)
    else:
        padw, padh = w-neww, h-newh
        padw_up = random.randint(0,padw)
        padh_left = random.randint(0,padh)
        padw_down = padw-padw_up
        padh_right = padh-padh_left
        
        trans_pad = transforms.Pad(padding=[padh_left, padw_down, padh_right, padw_up], fill=0, padding_mode='edge')
        img = trans_pad(img)
        mask = trans_pad(mask)
    
    return img, mask
'''

class CopyPaste(object):
    """
    "We randomly select two images and apply random scale jittering and random
    horizontal flipping on each of them. Then we select a random subset of objects
    from one of the images and paste them onto the other image."
    来自https://github.com/KimRass/Copy-Paste/blob/main/copy_paste.py
    """
    def __init__(self):
        pass

    @staticmethod
    def merge_two_images_using_mask(image1, image2, mask1, mask2, x=None, y=None):
        """
        "We compute the binary mask ($\alpha$) of pasted objects using ground-truth
        annotations and compute the new image as
        $I_{1} \times \alpha + I_{2} \times (1 - \alpha)$ where $I_{1}$ is
        the pasted image and $I_{2}$ is the main image."
        "To smooth out the edges of the pasted objects we apply a Gaussian filter
        to \alpha similar to “blending” in [13]. Simply composing without any blending has similar performance."
        """
        #!!!image2粘到image1上        
        new_img, new_mask = image1.clone(), mask1.clone()
        img_size_w,img_size_h = image1.size(1), image1.size(2)
        cutmix_w, cutmix_h = image2.size(1), image2.size(2)
        if x is None:
            while True:
                x = np.random.randint(0, img_size_w)
                y = np.random.randint(0, img_size_h)
                if x + cutmix_w <= img_size_w and y + cutmix_h <= img_size_h:
                    break
                
        new_img[:, x:x + cutmix_w, y:y + cutmix_h] = image2
        new_mask[x:x + cutmix_w, y:y + cutmix_h] = mask2
        
        return new_img, new_mask

    def apply(self, image1, annots1, image2, annots2, idx1, idx2):
        """
        "We remove fully occluded objects and update the masks and bounding boxes
        of partially occluded objects."
        """
        image11 = image1[idx1] #!!![3,h,w]
        mask11 = annots1[idx1] #!!![h,w]
        image12 = image2[idx2] #!!![3,h,w]
        mask12 = annots2[idx2] #!!![h,w]      
        
        labels = torch.unique(mask12)
        if torch.any(labels==0):
            labels = labels[1:]
        if torch.any(labels==255):
            labels = labels[:-1]       
        if labels.size(0) == 0:
            # return randomBright(image11, mask11)
            return adjustBright(image11, mask11)
        elif labels.size(0) == 1:
            select_label = labels.item()
        else:
            select_label = random.sample(labels.cpu().numpy().tolist(), 1)
            select_label=select_label[0]
            
        selected = (mask12 == select_label)
        mask12[selected] = 1
        mask12[~selected] = 0 
        
        num_fg = torch.sum(mask12)
        if num_fg == 0:
            # return randomBright(image11, mask11)
            return adjustBright(image11, mask11)
        
        bbox = get_bbox_from_mask(mask12, pad= 30)   
        flag = np.array(bbox)
        if (flag<0).any():
            return adjustBright(image11, mask11)
        cropped_data = crop_image_to_bbox(image12, bbox)
        cropped_seg = crop_seg_to_bbox(mask12, bbox).long()
        cropped_seg[cropped_seg>0] = 1 

        
        # image12, mask12 = adjustBright(cropped_data, cropped_seg)
        # image12, mask12 = randomRotate(image12, mask12)
        image12, mask12 = randomRotate(cropped_data, cropped_seg)
        
        x,y = bbox[0][0], bbox[1][0]
        new_image, new_mask = self.merge_two_images_using_mask(
                image1=image11, image2=image12, mask1=mask11, mask2=mask12
        )        
        new_mask = new_mask.long()
        
        return new_image, new_mask
    
    def __call__(self, image1, annots1, image2, annots2, batch_indices1=None, batch_indices2=None):
        if batch_indices1 is None:
            batch_size = image1.size(0)
            batch_indices1 = list(range(batch_size))
            batch_indices2 = random.sample(range(batch_size), batch_size)

        images = list()
        masks = list()
        for idx1, idx2 in zip(batch_indices1, batch_indices2):
            image_tensor, mask = self.apply(
                image1, annots1, image2, annots2, idx1, idx2)
            
            images.append(image_tensor)
            masks.append(mask)
            
        if not images:
            return
        
        return torch.stack(images, dim=0), torch.stack(masks, dim=0), batch_indices1, batch_indices2
#!!!--------------------------------