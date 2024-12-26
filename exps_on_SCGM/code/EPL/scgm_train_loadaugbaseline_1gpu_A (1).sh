#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

save_path=exp/scgm_train_loadaugbaseline/1gpu/test_A
mkdir -p $save_path

python scgm_train_loadaugbaseline_1gpu_A.py 2>&1 | tee $save_path/$now.log