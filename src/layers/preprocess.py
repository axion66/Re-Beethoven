import os
import torchaudio
import numpy as np
from typing import Optional, Union

import torch
from torch import Tensor as T
import torch.nn as nn
import torch.nn.functional as F

transform_book = {
    
}

def exists(val):
    return val is not None

def stereo_to_mono(waveform:T):
    '''
        waveform: [channel (1 for mono, 2 for stereo), length]
    '''
    if waveform.size(0) != 1:  
        return waveform.mean(dim=0).unsqueeze(0)
    return waveform  


def divide(waveform : T, n, cut_first):
    '''
        get [1, length] tensor
        turn into [1, length-cut_first:new_length],
        where the operation cuts the first cut_first length and make the rest divisible by n by cutting the last.
    '''
    waveform = waveform[:,cut_first:]
    _, length = waveform.shape
    new_length = length - (length % n)
    return waveform[:, :new_length]
     

def resample(waveform : T, orig_sr:int, new_sr:int):
    if (orig_sr == new_sr):
        return waveform
    
    key = f"{orig_sr}2{new_sr}"
    if key not in transform_book.keys():
        t = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=new_sr,
            lowpass_filter_width=64, 
            rolloff=0.94759371674, 
            resampling_method='sinc_interp_kaiser', 
            beta=14.76965645938
        )
        transform_book.update({key : t})
        
    return transform_book[key](waveform)

def load_audio(filepath:str,config):
    waveform, sr = torchaudio.load(filepath, normalize = True)      # normalize audio into [-1, 1]
    waveform = stereo_to_mono(waveform = waveform)
    waveform = resample(waveform = waveform, orig_sr = sr, new_sr = config['sr'])
    waveform = divide(waveform = waveform, n = config['hop_length'], cut_first = config['cut_first'])
    return waveform

def load_files(base_folder:str,config):
    audio_tensors = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".wav"):
                try:
                    file_path = os.path.join(root, file)
                    waveform = load_audio(file_path,config)
                    audio_tensors.append(waveform)
                except:
                    print(f"Error on passing audiofile {file}")
    
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

