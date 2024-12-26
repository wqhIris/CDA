import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2 

from dataloaders import utils
from dataloaders.dataset_covid import CovidDataSets
from networks.net_factory import net_factory
from utils import losses
from test_covid import get_model_metric

import os.path as osp
from networks.ema import ModelEMA
from utils.transform import obtain_cutmix_box,mix
import torchvision.utils as vutils
from utils.tumor_cp import tumor_cp_augmentation_transformV2

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/home/xjl/lx/xx_covid19/DA_covid/', help='Name of Experiment')
parser.add_argument('--root_path', type=str, default='/data/xjl/lx/xx_covid19/DA_covid/', help='Name of Experiment')
parser.add_argument('--labeled_per', type=float, default=0.1, help='percent of labeled data')
if True:
    parser.add_argument('--dataset_name', type=str, default='COVID249', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_0.1_l.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_0.1_u.xlsx', help='Name of dataset')
    # parser.add_argument('--tumor_dir', type=str, default="/home/xjl/lx/xx_covid19/DA_covid/data/COVID249/tumors_0.1/", help='tumor pool')
    parser.add_argument('--tumor_dir', type=str, default="/data/xjl/lx/xx_covid19/DA_covid/data/COVID249/tumors_0.1/", help='tumor pool')
    # parser.add_argument('--reload_path', type=str, default="/home/xjl/lx/xx_covid19/DA_covid/exp/COVID249/exp_baseline_tumorcp_0.1_unet/model_best.pth", help='pretrained model')
    # parser.add_argument('--reload_path', type=str, default="/home/xjl/lx/xx_covid19/DA_covid/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.1_unet/model_best.pth", help='pretrained model')
    parser.add_argument('--reload_path', type=str, default="/data/xjl/lx/xx_covid19/DA_covid/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.1_unet/model_best.pth", help='pretrained model')
else:
    parser.add_argument('--dataset_name', type=str, default='MOS1000', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_slice_label.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_slice_unlabel.xlsx', help='Name of dataset')

parser.add_argument('--exp', type=str, default='CDA_loadpretrain_re', help='experiment_name')

parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_epoch', type=int, default=40, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512], help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')

# label and unlabel
parser.add_argument('--batch_size_label', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--batch_size_unlabel', type=int, default=4, help='batch_size per gpu')

#!!!
parser.add_argument('--dataroot_path', type=str, default='/data/xjl/lx/xx_covid19/DA_covid/', help='path of data')
#!!!

args = parser.parse_args()


def obtain_bbox(batch_size, img_size):
    for i in range(batch_size):  
        if i == 0:
            MixMask = obtain_cutmix_box(img_size).unsqueeze(0)
        else:
            MixMask = torch.cat((MixMask, obtain_cutmix_box(img_size).unsqueeze(0)))
    return MixMask

def train(args, snapshot_path):
    base_lr = args.base_lr
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_epoch = args.max_epoch
    excel_file_name_label = args.excel_file_name_label
    excel_file_name_unlabel = args.excel_file_name_unlabel


    # create model
    student_model = net_factory(net_type=args.model)

    model_dict = student_model.state_dict()
    if args.reload_path:
        pretrained_dict = torch.load(args.reload_path) 
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        student_model.load_state_dict(model_dict)
    #!!!print('load '+args.reload_path)
    logging.info('load {}'.format(args.reload_path))


    ema_model = ModelEMA(student_model, 0.999)
    teacher_model = ema_model.ema

    # Define the dataset
    labeled_train_dataset = CovidDataSets(root_path=args.dataroot_path, dataset_name=args.dataset_name, file_name = excel_file_name_label, aug = True) #!!!args.root_path, dataset_name=args.dataset_name, file_name = excel_file_name_label, aug = True)
    unlabeled_train_dataset = CovidDataSets(root_path=args.dataroot_path, dataset_name=args.dataset_name, file_name = excel_file_name_unlabel, aug = True) #!!!args.root_path, dataset_name=args.dataset_name, file_name = excel_file_name_unlabel, aug = True)
    
    logging.info('The overall number of labeled training image equals to {}'.format(len(labeled_train_dataset)))
    logging.info('The overall number of unlabeled training images equals to {}'.format(len(unlabeled_train_dataset)))

    optimizer = optim.SGD(student_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    writer = SummaryWriter(snapshot_path + '/log')

    # Define the dataloader
    labeled_dataloader = DataLoader(labeled_train_dataset, batch_size = args.batch_size_label, shuffle = True, num_workers = 4, pin_memory = True)
    unlabeled_dataloader = DataLoader(unlabeled_train_dataset, batch_size = args.batch_size_unlabel, shuffle = True, num_workers = 4, pin_memory = True)
    
    max_dice = 0
    max_dice_ema = 0
    min_loss = 9999
    min_loss_ema = 9999
    #!!!
    min_loss1 = 9999
    min_loss_ema1 = 9999
    #!!!
    iter_num = 0
    max_iterations = max_epoch * len(unlabeled_dataloader)
    for epoch in range(max_epoch):
        epoch_loss = []
        epoch_loss_labeled = []
        epoch_loss_pseudo = []
        epoch_loss_pseudo_aug = []
        #!!!print("Start epoch ", epoch+1, "!")
        logging.info("Start epoch {} !".format(epoch+1))
        student_model.train()
        teacher_model.train()
        
        tbar = tqdm(range(len(unlabeled_dataloader)), ncols=70)
        labeled_dataloader_iter = iter(labeled_dataloader)
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)
        for batch_idx in tbar:
            try:
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()
            except StopIteration:
                labeled_dataloader_iter = iter(labeled_dataloader)
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()

            
            # labeled and unlabeled: weak view
            input_ul, target_ul, file_name_ul , lung_ul = unlabeled_dataloader_iter.next()
            input_ul, target_ul, lung_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True), lung_ul.cuda(non_blocking=True)
            input_l, target_l, lung_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True), lung_l.cuda(non_blocking=True)
            
            #!!!if input_l.shape[0]!=args.batch_size_label:
            #!!!    continue
            if input_l.shape[0]!=args.batch_size_label or input_ul.shape[0]!=args.batch_size_unlabel:
                continue
            

            # get pseudo labels from teacher_model for unlabeled data
            teacher_model.eval()
            with torch.no_grad():
                outputs_unlabeled  = teacher_model(input_ul)
                outputs_unlabeled_soft = torch.softmax(outputs_unlabeled, dim=1)
                pseudo_labels = torch.argmax(outputs_unlabeled_soft.detach(), dim=1, keepdim=False)
            teacher_model.train()


            #!!!            
            # Strong view: Cross-set data augmentation (cutmix)
            img_size = args.patch_size[1] #512
            bs1 = input_l.shape[0]
            bs2 = input_ul.shape[0]
            MixMask = obtain_bbox(bs1+bs2, img_size).cuda()
            # print('MixMask1.shape, input_l.shape, input_ul.shape:',MixMask1.shape, input_l.shape, input_ul.shape)
            input_aug1,rand_index1 = mix(MixMask[:bs1].unsqueeze(1).repeat(1, 3, 1, 1), input_l, input_ul) #l-->ul
            input_aug2,rand_index2 = mix(MixMask[bs1:bs1+bs2].unsqueeze(1).repeat(1, 3, 1, 1), input_ul, input_l) #ul-->l
            pseudo_labels_aug1,_ = mix(MixMask[:bs1], target_l, pseudo_labels,rand_index1)
            pseudo_labels_aug2,_ = mix(MixMask[bs1:bs1+bs2], pseudo_labels, target_l,rand_index2)
            
            
            # Strong view: Cross-set data augmentation (tumor cp)
            # labeled set--> labeled set
            input_l_aug, target_l_aug = tumor_cp_augmentation_transformV2(input_l, target_l, lung_l, args.tumor_dir)
            input_l_aug, target_l_aug = input_l_aug.cuda(non_blocking=True), target_l_aug.cuda(non_blocking=True)
            # labeled set--> unlabeled set
            input_ul_aug, pseudo_labels_aug = tumor_cp_augmentation_transformV2(input_ul, pseudo_labels, lung_ul, args.tumor_dir)
            input_ul_aug, pseudo_labels_aug = input_ul_aug.cuda(non_blocking=True), pseudo_labels_aug.cuda(non_blocking=True)
            
            # train student_model         
            volume_batch = torch.cat([input_l, input_l_aug, input_ul_aug, input_aug1, input_aug2], 0)
            outputs_l, outputs_l_aug, outputs_ul_aug, lambda1, lambda2 = student_model(volume_batch, mode='train', x2=input_ul, alpha=1.0)
            outputs = torch.cat([outputs_l, outputs_l_aug, outputs_ul_aug], 0)
            outputs_soft = torch.softmax(outputs, dim=1)
            label_batch_mix = lambda1*target_l + lambda2*pseudo_labels
            label_batch = torch.cat([target_l, target_l_aug, pseudo_labels_aug, pseudo_labels_aug1, pseudo_labels_aug2, label_batch_mix, label_batch_mix], 0)
            
            
            labeled_loss = (ce_loss(outputs[:args.batch_size_label], label_batch[:][:args.batch_size_label].long()) + dice_loss(
                outputs_soft[:args.batch_size_label], label_batch[:args.batch_size_label].unsqueeze(1)))
            pseudo_supervision =  (ce_loss(outputs[args.batch_size_label:args.batch_size_label*2], label_batch[:][args.batch_size_label:args.batch_size_label*2].long()) + dice_loss(
                outputs_soft[args.batch_size_label:args.batch_size_label*2], label_batch[args.batch_size_label:args.batch_size_label*2].unsqueeze(1)))            
            
            pseudo_supervision_aug1 = (ce_loss(outputs[args.batch_size_label*2:args.batch_size_label*3], label_batch[:][args.batch_size_label*2:args.batch_size_label*3].long()) + dice_loss(
                outputs_soft[args.batch_size_label*2:args.batch_size_label*3], label_batch[args.batch_size_label*2:args.batch_size_label*3].unsqueeze(1)))
            pseudo_supervision_aug2 = (ce_loss(outputs[args.batch_size_label*3:args.batch_size_label*4], label_batch[:][args.batch_size_label*3:args.batch_size_label*4].long()) + dice_loss(
                outputs_soft[args.batch_size_label*3:args.batch_size_label*4], label_batch[args.batch_size_label*3:args.batch_size_label*4].unsqueeze(1)))
            
            
            pseudo_supervision_aug3 =  (ce_loss(outputs[args.batch_size_label*4:args.batch_size_label*5], label_batch[:][args.batch_size_label*4:args.batch_size_label*5].long())+dice_loss(
                outputs_soft[args.batch_size_label*4:args.batch_size_label*5], label_batch[args.batch_size_label*4:args.batch_size_label*5].unsqueeze(1)))
            pseudo_supervision_aug4 =  (ce_loss(outputs[args.batch_size_label*5:args.batch_size_label*6], label_batch[:][args.batch_size_label*5:args.batch_size_label*6].long()) + dice_loss(
                outputs_soft[args.batch_size_label*5:args.batch_size_label*6], label_batch[args.batch_size_label*5:args.batch_size_label*6].unsqueeze(1)))
            pseudo_supervision_aug5 =  (ce_loss(outputs[args.batch_size_label*6:args.batch_size_label*7], label_batch[:][args.batch_size_label*6:args.batch_size_label*7].long()) + dice_loss(
                outputs_soft[args.batch_size_label*6:args.batch_size_label*7], label_batch[args.batch_size_label*6:args.batch_size_label*7].unsqueeze(1)))
            
            loss = labeled_loss + pseudo_supervision + pseudo_supervision_aug1 + pseudo_supervision_aug2 + pseudo_supervision_aug3  + pseudo_supervision_aug4 + pseudo_supervision_aug5


            
            #!!!
            epoch_loss.append(float(loss))
            epoch_loss_labeled.append(float(labeled_loss))
            epoch_loss_pseudo.append(float(pseudo_supervision))
            epoch_loss_pseudo_aug.append(float((1.0/6.0)*(pseudo_supervision+pseudo_supervision_aug1+pseudo_supervision_aug2+pseudo_supervision_aug3 + pseudo_supervision_aug4+pseudo_supervision_aug5)))
            #!!!
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model.update(student_model)
            teacher_model = ema_model.ema

            

            # if iter_num %100==0: 
            #     writer.add_image(str(iter_num)+'train/input_l', vutils.make_grid(input_l.detach(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     writer.add_image(str(iter_num)+'train/target_l', vutils.make_grid(target_l.unsqueeze(1).repeat(1, 3, 1, 1).detach().float(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     writer.add_image(str(iter_num)+'train/input_ul', vutils.make_grid(input_ul.detach(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     writer.add_image(str(iter_num)+'train/pseudo_labels', vutils.make_grid(pseudo_labels.unsqueeze(1).repeat(1, 3, 1, 1).detach().float(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     writer.add_image(str(iter_num)+'train/input_l_aug', vutils.make_grid(input_l_aug.detach(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     writer.add_image(str(iter_num)+'train/target_l_aug', vutils.make_grid(target_l_aug.unsqueeze(1).repeat(1, 3, 1, 1).detach().float(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     writer.add_image(str(iter_num)+'train/input_ul_aug', vutils.make_grid(input_ul_aug.detach(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     writer.add_image(str(iter_num)+'train/pseudo_labels_aug', vutils.make_grid(pseudo_labels_aug.unsqueeze(1).repeat(1, 3, 1, 1).detach().float(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # writer.add_scalar('info/total_loss', loss, iter_num)
            # logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

        '''epoch_loss = np.mean(epoch_loss)
        epoch_loss_labeled = np.mean(epoch_loss_labeled)
        epoch_loss_pseudo_aug = np.mean(epoch_loss_pseudo_aug)'''
        #!!!
        epoch_loss = np.mean(epoch_loss)
        epoch_loss_labeled = np.mean(epoch_loss_labeled)
        epoch_loss_pseudo = np.mean(epoch_loss_pseudo)
        epoch_loss_pseudo_aug = np.mean(epoch_loss_pseudo_aug)
        #!!!
        #!!!print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4},loss_labeled = {:.4},loss_pseudo_aug = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'], epoch_loss.item(),\
        #!!!    epoch_loss_labeled.item(),epoch_loss_pseudo_aug.item()))
        logging.info('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4},loss_labeled = {:.4}, loss_pseudo = {:.4}, loss_pseudo_aug = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'], epoch_loss.item(), epoch_loss_labeled.item(), epoch_loss_pseudo.item(), epoch_loss_pseudo_aug.item()))
               
            
        #evaluation
        if (epoch >= 0):
            student_model.eval()
            teacher_model.eval()
            with torch.no_grad():
                #!!!print('epoch:'+str(epoch))
                logging.info('epoch:{}'.format(epoch))
                # nsd, dice = get_model_metric(args, student_model, snapshot_path, args.model, mode='val')
                # print('student model--nsd dice:'+str(nsd)+'    '+str(dice))
                val_loss = evaluation(args, student_model, snapshot_path, args.model, ce_loss, dice_loss, mode='val')
                #!!!print('val_loss:',val_loss)
                logging.info('val_loss:{}'.format(val_loss))
                #!!!
                val_loss1 = evaluation(args, student_model, snapshot_path, args.model, ce_loss, dice_loss, mode='test')
                #!!!print('test_loss:',val_loss1)
                logging.info('test_loss:{}'.format(val_loss1))
                #!!!
                
                # nsd_ema, dice_ema = get_model_metric(args, teacher_model, snapshot_path, args.model, mode='val')
                # print('student model--nsd dice:'+str(nsd_ema)+'    '+str(dice_ema))
                # writer.add_scalars('evaluation_dice', {'student_dice':dice, 'teacher_dice':dice_ema}, epoch)
                val_loss_ema = evaluation(args, teacher_model, snapshot_path, args.model, ce_loss, dice_loss, mode='val')
                #!!!print('val_loss_ema:',val_loss_ema)
                logging.info('val_loss_ema:{}'.format(val_loss_ema))
                #!!!
                val_loss_ema1 = evaluation(args, teacher_model, snapshot_path, args.model, ce_loss, dice_loss, mode='test')
                #!!!print('test_loss_ema:',val_loss_ema1)
                logging.info('test_loss_ema:{}'.format(val_loss_ema1))
                #!!!
                  
                # if dice > max_dice:
                if val_loss <= min_loss:
                    torch.save(student_model.state_dict(), osp.join(snapshot_path, 'model_best.pth'))
                    # max_dice = dice
                    min_loss = val_loss
                    #!!!print("=> saved student model")
                    logging.info("=> saved student model")

                # if dice_ema > max_dice_ema:
                if val_loss_ema <= min_loss_ema:
                    torch.save(teacher_model.state_dict(), osp.join(snapshot_path, 'ema_model_best.pth'))
                    # max_dice_ema = dice_ema
                    min_loss_ema = val_loss_ema
                    #!!!print("=> saved teacher model")
                    logging.info("=> saved teacher model")

                #!!!
                if val_loss1 <= min_loss1:
                    torch.save(student_model.state_dict(), osp.join(snapshot_path, 'model_best_test.pth'))
                    # max_dice = dice
                    min_loss1 = val_loss1
                    #!!!print("=> saved student model test")
                    logging.info("=> saved student model test")

                # if dice_ema > max_dice_ema:
                if val_loss_ema1 <= min_loss_ema1:
                    torch.save(teacher_model.state_dict(), osp.join(snapshot_path, 'ema_model_best_test.pth'))
                    # max_dice_ema = dice_ema
                    min_loss_ema1 = val_loss_ema1
                    #!!!print("=> saved teacher model test")
                    logging.info("=> saved teacher model test")
                #!!!
                
                # print("best val dice:{0}".format(max_dice)) 
                # print("best val dice_ema:{0}".format(max_dice_ema)) 
                #!!!print("best val loss:{0}".format(min_loss)) 
                #!!!print("best val loss_ema:{0}".format(min_loss_ema)) 
                logging.info("best val loss:{0}".format(min_loss))
                logging.info("best val loss_ema:{0}".format(min_loss_ema)) 
                #!!!
                #!!!print("best test loss:{0}".format(min_loss1)) 
                #!!!print("best test loss_ema:{0}".format(min_loss_ema1)) 
                logging.info("best test loss:{0}".format(min_loss1))
                logging.info("best test loss_ema:{0}".format(min_loss_ema1)) 
                #!!!
                
    writer.close()
def evaluation(args, model, snapshot_path, model_name, ce_loss, dice_loss, mode='val'):

    model.eval()
    file_slice_name = '{}_slice.xlsx'.format(mode)
    file_volume_name = '{}_volume.xlsx'.format(mode)
    val_dataset = CovidDataSets(root_path=args.dataroot_path, dataset_name=args.dataset_name, file_name = file_slice_name) #!!!args.root_path, dataset_name=args.dataset_name, file_name = file_slice_name)
    #!!!print('The overall number of validation images equals to %d' % len(val_dataset))
    logging.info('The overall number of validation images equals to {}'.format(len(val_dataset)))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_loss = 0.0
    
    for batch_idx, (image, label, file_name, _) in enumerate(val_dataloader):
        image = image.cuda()
        label = label.cuda()

        with torch.no_grad():
            outputs = model(image)
            outputs_soft = torch.softmax(outputs, dim=1)
            loss = (ce_loss(outputs, label.long()) + dice_loss(outputs_soft, label.unsqueeze(1)))
            val_loss += loss.item()
    return val_loss/len(val_dataloader)

if __name__ == "__main__":
    # for reproducing
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    seed = 66
    #!!!print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    snapshot_path = "{}exp/{}/exp_{}_{}_{}".format(args.root_path, args.dataset_name, args.exp, args.labeled_per, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    #!!!
    logging.info("[ Using Seed : {}]".format(seed))
    #!!!
    
    train(args, snapshot_path)