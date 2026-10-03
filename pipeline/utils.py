import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets random seeds across Python, NumPy, PyTorch (CPU and CUDA),
    and sets cuDNN deterministic mode to guarantee reproducible runs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seeded_generator(seed: int = 42) -> torch.Generator:
    """Returns a PyTorch Generator initialized with the specified seed for DataLoaders."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int):
    """Worker initialization function for multi-processing PyTorch DataLoaders."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
