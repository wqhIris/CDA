from networks.unet2d import UNet2D
from networks.unet2d_imgcut_feaaugmixup import UNet2D as unet_imgcut_feaaugmixup

def net_factory(net_type="unet"):
    if net_type == "unet":
        net = UNet2D().cuda()
    ##!!!
    elif net_type == 'unet_imgcut_feaaugmixup':
        net = unet_imgcut_feaaugmixup().cuda()
    #!!!
    else:
        net = None 
    return net