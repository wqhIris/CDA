#cd /opt/data/private/semi-medical/codes/ours/code/

#pip install -r /mnt/DA_covid/requirements.txt

#--------baseline-----------------
#train
CUDA_VISIBLE_DEVICES=0 python train_baseline.py --root_path /root/autodl-fs/ --dataroot_path /root/autodl-fs/ --exp baseline_0.1ratio --excel_file_name_label train_0.1_l.xlsx --excel_file_name_unlabel train_0.1_u.xlsx --labeled_per 0.1
#CUDA_VISIBLE_DEVICES=0 python train_baseline.py --root_path /root/autodl-fs/ --dataroot_path /root/autodl-fs/ --exp baseline_0.2ratio --excel_file_name_label train_0.2_l.xlsx --excel_file_name_unlabel train_0.2_u.xlsx --labeled_per 0.2
#CUDA_VISIBLE_DEVICES=0 python train_baseline.py --root_path /root/autodl-fs/ --dataroot_path /root/autodl-fs/  --exp baseline_0.3ratio --excel_file_name_label train_0.3_l.xlsx --excel_file_name_unlabel train_0.3_u.xlsx --labeled_per 0.3

#test
CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_baseline_0.1ratio_0.1_unet/model_best.pth --dataroot_path /root/autodl-fs/ --exp baseline_0.1ratio --labeled_per 0.1
#CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_baseline_0.2_unet/model_best.pth --dataroot_path /root/autodl-fs/ --exp baseline_0.2ratio --labeled_per 0.2
#CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_baseline_0.3_unet/model_best.pth --dataroot_path /root/autodl-fs/ --exp baseline_0.3ratio --labeled_per 0.3
#--------






#--------BCP-----------------
#train
CUDA_VISIBLE_DEVICES=0 python train_BCP.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.1/ --reload_path /root/autodl-fs/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.1_unet/model_best.pth  --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.1_l.xlsx --excel_file_name_unlabel train_0.1_u.xlsx --exp BCP --labeled_per 0.1
#CUDA_VISIBLE_DEVICES=0 python train_BCP.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.2/ --reload_path /root/autodl-fs/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.2_unet/model_best.pth  --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.2_l.xlsx --excel_file_name_unlabel train_0.2_u.xlsx --exp BCP --labeled_per 0.2
#CUDA_VISIBLE_DEVICES=0 python train_BCP.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.3/ --reload_path /root/autodl-fs/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.3_unet/model_best.pth  --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.3_l.xlsx --excel_file_name_unlabel train_0.3_u.xlsx --exp BCP --labeled_per 0.3

#test
CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_BCP_0.1_unet/model_best.pth --dataroot_path /root/autodl-fs/ --exp BCP --labeled_per 0.1
#CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_BCP_0.2_unet/model_best.pth --dataroot_path /root/autodl-fs/ --exp BCP --labeled_per 0.2
#CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_BCP_0.3_unet/model_best.pth --dataroot_path /root/autodl-fs/ --exp BCP --labeled_per 0.3
#--------







#--------ours CDA-----------------
#--------step1: pretrain using tumor copy-paste-blur
#1)extract all tumors from labeled training dimage for lession copy-paste (need change parameters: data_root,file_path,save_dir for different labeled ratio) -----------------
python extract_tumors.py
#2)baseline + tumor copy-paste-blur-----------------
#train
CUDA_VISIBLE_DEVICES=0 python train_baseline_tumorcptransV2.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.1/ --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.1_l.xlsx --excel_file_name_unlabel train_0.1_u.xlsx --exp baseline_tumorcp_transform_blur --labeled_per 0.1
#CUDA_VISIBLE_DEVICES=0 python train_baseline_tumorcptransV2.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.2/ --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.2_l.xlsx --excel_file_name_unlabel train_0.2_u.xlsx --exp baseline_tumorcp_transform_blur --labeled_per 0.2
#CUDA_VISIBLE_DEVICES=0 python train_baseline_tumorcptransV2.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.3/ --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.3_l.xlsx --excel_file_name_unlabel train_0.3_u.xlsx --exp baseline_tumorcp_transform_blur --labeled_per 0.3

#--------step2: train using CDA and load pretrained params
#train
CUDA_VISIBLE_DEVICES=0 python train_CDA.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.1/ --reload_path /root/autodl-fs/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.1_unet/model_best.pth  --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.1_l.xlsx --excel_file_name_unlabel train_0.1_u.xlsx --exp CDA --model unet_imgcut_feaaugmixup --labeled_per 0.1
#CUDA_VISIBLE_DEVICES=0 python train_CDA.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.2/ --reload_path /root/autodl-fs/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.2_unet/model_best.pth  --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.2_l.xlsx --excel_file_name_unlabel train_0.2_u.xlsx --exp CDA --model unet_imgcut_feaaugmixup --labeled_per 0.2
#CUDA_VISIBLE_DEVICES=0 python train_CDA.py --root_path /root/autodl-fs/ --tumor_dir /root/autodl-fs/data/COVID249/tumor_0.3/ --reload_path /root/autodl-fs/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.3_unet/model_best.pth  --dataroot_path /root/autodl-fs/ --excel_file_name_label train_0.3_l.xlsx --excel_file_name_unlabel train_0.3_u.xlsx --exp CDA --model unet_imgcut_feaaugmixup --labeled_per 0.3


#test
CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_CDA_0.1_unet_imgcut_feaaugmixup/model_best.pth --dataroot_path /root/autodl-fs/ --exp CDA --model unet_imgcut_feaaugmixup --labeled_per 0.1
#CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_CDA_0.2_unet_imgcut_feaaugmixup/model_best.pth --dataroot_path /root/autodl-fs/ --exp CDA --model unet_imgcut_feaaugmixup --labeled_per 0.2
#CUDA_VISIBLE_DEVICES=0 python test.py --root_path /root/autodl-tmp/ --model_path /root/autodl-fs/exp/COVID249/exp_CDA_0.3_unet_imgcut_feaaugmixup/model_best.pth --dataroot_path /root/autodl-fs/ --exp CDA --model unet_imgcut_feaaugmixup --labeled_per 0.3
#-------- 