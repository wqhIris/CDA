#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=exp/baseline_instancecp_deeplabv3_epldiceloss_savebestema_counttime/1gpu/test_C
mkdir -p $save_path

python train_baseline_instancecp_deeplabv3_epldiceloss_savebestema_C.py 2>&1 | tee $save_path/$now.log