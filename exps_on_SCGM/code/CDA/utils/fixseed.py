import torch
import numpy as np
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  #!!!设置为True的话效率会低
    torch.backends.cudnn.deterministic=True  #!!!设置为True的话，确保每次返回的卷积算法将是确定的