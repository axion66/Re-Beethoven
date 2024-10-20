import os
import torchaudio
import numpy as np
from typing import Optional, Union

import torch
from torch import Tensor as T
import torch.nn as nn
import torch.nn.functional as F


def exists(val):
    return val is not None

def stereo_to_mono(waveform:T,dim=-1):
    '''
        waveform: [channel (1 for mono, 2 for stereo), length]
    '''
    if waveform.size(0) == 2:  
        waveform = waveform.mean(dim=0).unsqueeze(0)
    return waveform  

def divide(tensor:T,n,cut_first):
    '''
        get [1, length] tensor
        turn into [1, length-cut_first_n:new_length]
    '''
    tensor = tensor[:,cut_first:]
    _, length = tensor.shape
    new_length = length - (length % n)
    return tensor[:, :new_length]
     
     
def resample(waveform:T, orig_sr:int,new_sr:int,kaisier_best=True):
    # TODO: better to use torchaudio.transforms.resample for caching. [if all orig_sr are the same]
    if kaisier_best:
        return torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=new_sr,lowpass_filter_width=64, \
                                            rolloff=0.94759371674,resampling_method="sinc_interp_kaiser",beta=14.76965645938)
    return torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=new_sr,lowpass_filter_width=32)

def load_audio(filepath:str,config):
    waveform, sr = torchaudio.load(filepath)
    waveform = stereo_to_mono(waveform,dim=-1)
    waveform = resample(waveform,sr,config['sr'])
    waveform = divide(waveform,n=config['hop_length'],cut_first=config['cut_first'])
    return waveform

def load_mp3_files(base_folder:str,config):
    audio_tensors = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".mp3" or ".wav"):
                file_path = os.path.join(root, file)
                waveform = load_audio(file_path,config)
                audio_tensors.append(waveform)
    
    
    return audio_tensors

def create_overlapping_chunks_tensor(sequence: torch.Tensor, config) -> torch.Tensor:

    if sequence.dim() != 2 or sequence.size(0) != 1:
        raise ValueError("Input tensor must have shape (1, sequence_length)")

    sequence = sequence.flatten() 
    
    step_size = config['seq_len'] // config['overlap_ratio']
    num_chunks = (sequence.size(0) - config['seq_len']) // step_size + 1
    
    chunks = torch.zeros((num_chunks, config['seq_len']))

    for i in range(num_chunks):
        start_idx = i * step_size
        chunks[i] = sequence[start_idx:start_idx + config['seq_len']]
    
    return chunks

