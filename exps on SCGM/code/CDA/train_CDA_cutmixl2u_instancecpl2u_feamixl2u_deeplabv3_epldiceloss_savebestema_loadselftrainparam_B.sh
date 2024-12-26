#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=exp/cutmixl2u_instancecpl2u_feamixl2u_deeplabv3_epldiceloss_savebestema_loadselftrainparam/1gpu/test_B
mkdir -p $save_path

python train_CDA_cutmixl2u_instancecpl2u_feamixl2u_deeplabv3_epldiceloss_savebestema_loadselftrainparam_B.py 2>&1 | tee $save_path/$now.log