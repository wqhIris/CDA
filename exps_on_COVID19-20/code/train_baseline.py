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

from dataloaders import utils
from dataloaders.dataset_covid import CovidDataSets
from networks.net_factory import net_factory
from utils import losses
import cv2
import os.path as osp

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/home/lx/xx_covid19/SASSL-master/', help='Name of Experiment')
parser.add_argument('--root_path', type=str, default='/data/xjl/lx/xx_covid19/DA_covid/', help='Name of Experiment')

parser.add_argument('--consistency_syn', type=float, default=0.5, help='consistency')
parser.add_argument('--consistency_pseudo', type=float, default=0.5, help='consistency')

# parser.add_argument('--labeled_per', type=float, default=0.1, help='percent of labeled data')
parser.add_argument('--labeled_per', type=float, default=0.3, help='percent of labeled data')
if True:
    parser.add_argument('--dataset_name', type=str, default='COVID249', help='Name of dataset')
    # parser.add_argument('--excel_file_name_label', type=str, default='train_0.1_l.xlsx', help='Name of dataset')
    # parser.add_argument('--excel_file_name_unlabel', type=str, default='train_0.1_u.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_0.3_l.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_0.3_u.xlsx', help='Name of dataset')
else:
    parser.add_argument('--dataset_name', type=str, default='MOS1000', help='Name of dataset')
    parser.add_argument('--excel_file_name_label', type=str, default='train_slice_label.xlsx', help='Name of dataset')
    parser.add_argument('--excel_file_name_unlabel', type=str, default='train_slice_unlabel.xlsx', help='Name of dataset')

parser.add_argument('--exp', type=str, default='baseline_0.3ratio', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_epoch', type=int, default=40, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512], help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--batch_size_label', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--batch_size_unlabel', type=int, default=8, help='batch_size per gpu')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=10.0, help='consistency_rampup')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha')

#!!!
parser.add_argument('--dataroot_path', type=str, default='/data/xjl/lx/xx_covid19/DA_covid/', help='path of data')
#!!!

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    excel_file_name_label = args.excel_file_name_label
    excel_file_name_unlabel = args.excel_file_name_unlabel


    # create model
    model1 = net_factory(net_type=args.model)
    
    # Define the dataset
    labeled_train_dataset = CovidDataSets(root_path=args.dataroot_path, dataset_name=args.dataset_name, file_name = excel_file_name_label, aug = True) #!!!args.root_path, dataset_name=args.dataset_name, file_name = excel_file_name_label, aug = True)
    # print('The overall number of unlabeled training images equals to %d' % len(unlabeled_train_dataset))
    logging.info('The overall number of labeled training image equals to {}'.format(len(labeled_train_dataset)))


    # start training
    model1.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    #logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance1 = 0.0
    best_performance2 = 0.0

    # Define the dataloader
    labeled_dataloader = DataLoader(labeled_train_dataset, batch_size = args.batch_size_label, shuffle = True, num_workers = 4, pin_memory = True)
    max_iterations = max_epoch * len(labeled_dataloader)

    max_dice = 0
    min_loss = 9999
    min_loss1 = 9999 #!!!
    for epoch in range(max_epoch):
        epoch_loss = []
        #!!!print("Start epoch ", epoch, "!")
        logging.info("Start epoch {} !".format(epoch+1))
        
        tbar = tqdm(range(len(labeled_dataloader)), ncols=70)
        labeled_dataloader_iter = iter(labeled_dataloader)
        # unlabeled_dataloader_iter = iter(unlabeled_dataloader)

        for batch_idx in tbar:
            try:
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()
            except StopIteration:
                labeled_dataloader_iter = iter(labeled_dataloader)
                input_l, target_l, file_name_l , lung_l = labeled_dataloader_iter.next()
                print('length: style_output_global_positive_list')
                print(len(style_output_global_positive_list))

                style_output_global_positive_list =[]
                style_output_global_list =[]
        
           
            input_l, target_l, lung_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True), lung_l.cuda(non_blocking=True)

            if input_l.shape[0]!=args.batch_size_label:
                continue

            
            volume_batch = torch.cat([input_l], 0)
            label_batch = torch.cat([target_l], 0)

            outputs_1 = model1(volume_batch)
            outputs_soft_1 = torch.softmax(outputs_1, dim=1)

            labeled_loss_1 = (ce_loss(outputs_1[:args.batch_size_label], label_batch[:][:args.batch_size_label].long()) + dice_loss(
                outputs_soft_1[:args.batch_size_label], label_batch[:args.batch_size_label].unsqueeze(1)))

            model1_loss = labeled_loss_1
            loss = model1_loss
            epoch_loss.append(float(loss))

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            # write summary
            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            # logging.info('iteration %d : model1 loss : %f' % (iter_num, model1_loss.item()))

        epoch_loss = np.mean(epoch_loss)
        #!!!print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer1.param_groups[0]['lr'], epoch_loss.item()))
        logging.info('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer1.param_groups[0]['lr'], epoch_loss.item()))
            
        #evaluation
        if (epoch >= 0):
            model1.eval()
            with torch.no_grad():
                #!!!print('epoch:'+str(epoch))
                logging.info('epoch:{}'.format(epoch))
                # nsd, dice = get_model_metric(args, model1, snapshot_path, args.model, mode='val')
                # print('model--nsd dice:'+str(nsd)+'    '+str(dice))
                val_loss = evaluation(args, model1, snapshot_path, args.model, ce_loss, dice_loss, mode='val')
                #!!!print('val_loss:',val_loss)
                logging.info('val_loss:{}'.format(val_loss))
                #!!!
                
                
                # if dice > max_dice:
                if val_loss <= min_loss:
                    torch.save(model1.state_dict(), osp.join(snapshot_path, 'model_best.pth'))
                    # max_dice = dice
                    min_loss = val_loss
                    #!!!print("=> saved model")
                    logging.info("=> saved model")
                    
                # print("best val dice:{0}".format(max_dice)) 
                #!!!print("best val loss:{0}".format(min_loss)) 
                logging.info("best val loss:{0}".format(min_loss))


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
