ratio_pre = 0.2
test_vendor_pre = 'B'
name = 'SCGM_fixmatch_ratio'+str(ratio_pre)+'_'+test_vendor_pre+'_CM'
# hyperparameter
default_config = dict(
    batch_size=8, #!!!8, #!!!16, #!!!4, #!!!8, #!!!32,
    num_epoch=50,
    learning_rate=0.001, #!!!0.01, #!!!0.001, #!!!5e-5, #!!!1e-4,            # learning rate of Adam
    weight_decay=0.01, #!!!0.1, #!!!0.001, #!!!0.01, #!!!0.0001, #!!!0.01, #!!!0.1, #!!!0.01,             # weight decay 
    num_workers=8,

    train_name = name,
    model_path = name+'.pt',
    test_vendor = test_vendor_pre,
    ratio = ratio_pre,                   # 2%
    #!!!CPS_weight = 1.5, #!!!3, #!!!公式9的beta系数
    gpus= [0], #!!!, 1 , 2, 3],
    ifFast = False,
    Pretrain =False,
    Loadselftrain = True,
    pretrain_file = '/root/autodl-fs/exps_on_SCGM/CDA/tmodel_scgm/1gpu/baseline_instance_deeplabv3_epldiceloss_savebestema/lr0.001_noloadresnetparam_changecopypastev2_changeconstrast_randomrotatepaste_test2/stu_SCGM_fixmatch_ratio0.2_B_CM.pt', #'',#'resnet50_v1c.pth', #!!!'/home/hyaoad/remote/semi_medical/MNMS_seg/pretrain_res/resnet50_v1c.pth',

    
    #!!!-----------------------好像用不上
    restore = False,
    restore_from = name+'.pt',

    # for cutmix
    cutmix_mask_prop_range = (0.25, 0.5),
    cutmix_boxmask_n_boxes = 3,
    cutmix_boxmask_fixed_aspect_ratio = True,
    cutmix_boxmask_by_size = True,
    cutmix_boxmask_outside_bounds = True,
    cutmix_boxmask_no_invert = True,
    #!!!-----------------------好像用不上
    
    
    Fourier_aug = True,
    fourier_mode = 'AS',

    # for bn
    bn_eps = 1e-5,
    bn_momentum = 0.1,
)
