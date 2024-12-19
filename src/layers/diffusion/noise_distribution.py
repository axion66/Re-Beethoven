import torch
import torch.nn as nn
import numpy as np
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normal_noise_like(x):
  b, c, l = x.shape
  return torch.randn_like(x).to(DEVICE)

def pyramid_noise_like(x):
  '''
    Noise that contains low-frequency components as well as high-frequency ones.
  '''
  discount = 0.6
  b, c, l = x.shape 
  u = nn.Upsample(size=(l), mode='bilinear')
  noise = torch.randn_like(x)
  for i in range(10):
    r = random.random()*2+2 
    l = max(1, int(l/(r**i)))
    noise += u(torch.randn(b, c, l).to(x)) * discount ** i
    if l==1: break 
  return (noise / noise.std()).to(DEVICE)  # make std of 1.

