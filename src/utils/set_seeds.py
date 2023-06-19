import random
import os
import numpy as np
import torch
import accelerate


def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    accelerate.utils.set_seed(seed)
