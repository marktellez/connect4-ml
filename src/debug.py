import random
import torch
import numpy as np

DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def set_seed():
    seed_value = 420
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy
    torch.manual_seed(seed_value)  # PyTorch

    # If you are using CUDA (PyTorch with GPU)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
