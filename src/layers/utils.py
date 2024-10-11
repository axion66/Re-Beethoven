import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def exists(val):
    return val is not None


def get_activation_fn(activation):
    if activation == "relu":
        activation_fn = nn.ReLU
    elif activation == "swish":
        activation_fn = nn.SiLU
    else:
        raise NotImplementedError
    return activation_fn
