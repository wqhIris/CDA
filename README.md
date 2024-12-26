# CDA
This repository contains the official PyTorch implementation of the paper **Cross-Set Data Augmentation for Semi-Supervised Medical Image Segmentation**

# Abstract
Medical image semantic segmentation is a fundamental yet challenging research task. However, training a fully supervised model for this task requires a substantial amount of pixel-level annotated data, which poses a significant challenge for annotators due to the necessity of specialized medical expert knowledge. To mitigate the labeling burden, a semi-supervised medical image segmentation model that leverages both a small quantity of labeled data and a substantial amount of unlabeled data has attracted prominent attention. However, the performance of current methods is constrained by the distribution mismatch problem between limited labeled and unlabeled datasets. To address this issue, we propose a cross-set data augmentation strategy aimed at minimizing the feature divergence between labeled and unlabeled data. Our approach involves mixing labeled and unlabeled data, as well as integrating ground truth with pseudo-labels to produce augmented samples. By employing three distinct cross-set data augmentation strategies, we enhance the diversity of the training dataset and fully exploit the perturbation space. Our experimental results on COVID-19 CT data, pinal cord gray matter MRI data and prostate T2-weighted MRI data substantiate the efficacy of our proposed approach.

# The overall framework
<img src="https://raw.githubusercontent.com/wqhIris/CDA/master/framework.png" width="482" height="343" alt="framework">

# How to install
## 1. Environment configuration
- We recommend installing the environment through conda and pip
- For experiments on COVID-19-20 dataset
```bash
cd CDA-main/exps_on_COVID19-20/code/

# Create a conda environment (python version >= 3.8)
conda create -n covid python=3.8

# Activate the environment
conda activate covid

# Install torch (torch version >= 1.11.0, cuda version >= 1.13) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies
pip install -r requirements.txt
```

- For experiments on SCGM dataset
```bash
cd CDA-main/exps_on_SCGM/code/

# Install dependencies and conda environment 
conda env create -f semi_dg_cda.yaml

# Activate the environment
conda activate semidg_cda
```

## 2. Data preparation
- COVID-19-20 
  - We followed the settings of [SASSL](https://github.com/FeiLyu/SASSL/) and the pre-processed dataset can be downloaded from the [link](https://drive.google.com/file/d/1A2f3RRblSByFncUlf5MEr9VEjFlqD0ge/view?usp=sharing).
  - Extract the sets to `$ROOT/exps_on_COVID19-20/data`. The directory structure should look like as follows:
```bash
$ROOT/exps_on_COVID19-20/data/
├── COVID249/
│   ├── NII/ (Original dataset in NIFTI)
│   ├── PNG/ (Pre-processed dataset in PNG)
│   ├── train_0.1_l.xlsx (Datasplit for 10% setting)
│   ├── train_0.1_u.xlsx (Datasplit for 10% setting)
│   ├── train_0.2_l.xlsx (Datasplit for 20% setting)
│   ├── train_0.2_u.xlsx (Datasplit for 20% setting)
│   ├── train_0.3_l.xlsx (Datasplit for 30% setting)
│   ├── train_0.3_u.xlsx (Datasplit for 30% setting)
│   ├── test_slice.xlsx (Datasplit for testing)
│   ├── val_slice.xlsx (Datasplit for validation)
```

- SCGM
  - We followed the settings of [EPL](https://github.com/XMed-Lab/EPL_SemiDG) and the original dataset can be downloaded from the [official website](http://niftyweb.cs.ucl.ac.uk/challenge/index.php).
  - Extract the training and testing data to `$ROOT/exps_on_SCGM/data/scgm_rawdata/train` and `$ROOT/exps_on_SCGM/data/scgm_rawdata/test`, respectively.
  - You need first to change the dirs (lines 32 to 53) in the scripts [exps_on_SCGM/data/preprocess/save_SCGM_2D.py](exps_on_SCGM/data/preprocess/save_SCGM_2D.py), and then run `save_SCGM_2D.py` to split the original dataset into labeled and unlabeled sets in four domains.
  - The directory structure should look like as follows:
```bash
$ROOT/exps_on_SCGM/data/
├── scgm/
│   ├── scgm_split_2D_data/ (Pre-processed image in NUMPY ARRAYS)
│       ├── Labeled/ (Labeled data in four domains)
│           ├── vendorA/ (Datasplit for domain UCL)
│               ├── 000000.npz
│               ├── ...
│           ├── vendorB/ (Datasplit for domain Montreal)
│           ├── vendorC/ (Datasplit for domain Zurich)
│           ├── vendorD/ (Datasplit for domain Vanderbilt)
│       ├── Unlabeled/ (Unlabeled data in four domains)
│   ├── scgm_split_2D_mask/ (Pre-processed segmentation masks in NUMPY ARRAYS)
│       ├── Labeled/ (Labeled masks in four domains)
```

# How to run
## 1. Training
- For experiments on COVID-19-20
  - To train the `Baseline`, `BCP`, and our `CDA` with 10%, 20%, 30% with labeled ratios, please refer to [exps_on_COVID19-20/code/main.sh](exps_on_COVID19-20/code/main.sh)
```bash
python extract_tumors.py

CUDA_VISIBLE_DEVICES=0 python train_baseline_tumorcptransV2.py \
--root_path /root/autodl-fs/ \
--tumor_dir /root/autodl-fs/data/COVID249/tumor_0.1/ \
--dataroot_path /root/autodl-fs/ \
--excel_file_name_label train_0.1_l.xlsx \
--excel_file_name_unlabel train_0.1_u.xlsx \
--exp baseline_tumorcp_transform_blur \
--labeled_per 0.1

CUDA_VISIBLE_DEVICES=0 python train_CDA.py \
--root_path /root/autodl-fs/CDA-main/exps_on_COVID19-20 \
--tumor_dir /root/autodl-fs/data/COVID249/tumor_0.1/ \
--reload_path /root/autodl-fs/exp/COVID249/exp_baseline_tumorcp_transform_blur_0.1_unet/model_best.pth \
--dataroot_path /root/autodl-fs/ \
--excel_file_name_label train_0.1_l.xlsx \
--excel_file_name_unlabel train_0.1_u.xlsx \
--exp CDA \
--model unet_imgcut_feaaugmixup \
--labeled_per 0.1
```

- For experiments on SCGM
  - To train the `EPL` with 20% labeled ratios in four domains, please refer to the bash scripts in `exps_on_SCGM/code/EPL`

  - To train our `CDA` with same , please refer to the bash scripts in `exps_on_SCGM/code/CDA` 

## 2. Testing
- For experiments on COVID-19-20
  - To run the evaluation code, please refer to [exps_on_COVID19-20/code/main.sh](exps_on_COVID19-20/code/main.sh)
- For experiments on SCGM
  - To evaluate `EPL`  

# Acknowledgement
This repository is based on the codes and datasets provided by the projects: [SASSL](https://github.com/FeiLyu/SASSL/) and [EPL](https://github.com/XMed-Lab/EPL_SemiDG).

Thanks a lot for their great works.

# Citation
If you use this code in your research, please kindly cite the following papers:

```bibtex
@article{wu2024cross,
  title={Cross-Set Data Augmentation for Semi-Supervised Medical Image Segmentation},
  author={Wu, Qianhao and Jiang, Xixi and Zhang, Dong and Feng, Yifei and Tang, Jinhu},
  journal={Image and Vision Computing},
  year={2024}
}
```


