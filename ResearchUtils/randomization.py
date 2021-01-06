import torch
import numpy as np


def force_determinism(is_deterministic=False):
    if not is_deterministic:
        return
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)
