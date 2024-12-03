import numpy as np
import torch

def _ensure_numpy(data):
    """Ensure data is a numpy array on CPU"""
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)