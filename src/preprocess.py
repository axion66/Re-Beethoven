import os
import torchaudio
import numpy as np
from typing import Optional, Union
from utils import exists

import torch
from torch import Tensor as T
import torch.nn as nn
import torch.nn.functional as F


# config
SR = 8000
N_FFT = 128
WIN_LEN = 128
HOP_LENGTH = 64
CHUNK_LENGTH = 30
N_MELS = 64
MIN=1e-7
MAX=2e+5

# default sampling rate for piano. not even close w/ Nyquist[44100Khz]
'''
resampler = {}
resampler[44100] = Resampler(input_sr=44100, output_sr=SR, dtype=torch.float32)
resampler[48000] = Resampler(input_sr=48000, output_sr=SR, dtype=torch.float32)
'''


def stereo_to_mono(waveform:T,dim=-1):
    '''
        waveform: [channel (1 for mono, 2 for stereo), length]
    '''
    if waveform.size(0) == 2:  
        waveform = waveform.mean(dim=0).unsqueeze(0)
    return waveform  

def divide(tensor:T,n,cut_firstlast_n):
    '''
        get [1, length] tensor
        turn into [1, length-cut_first_n:new_length]
    '''
    tensor = tensor[:,cut_firstlast_n:-cut_firstlast_n]
    _, length = tensor.shape
    new_length = length - (length % n)
    return tensor[:, :new_length]
     
     
def resample(waveform:T, orig_sr:int,new_sr:int,kaisier_best=True):
    # TODO: better to use torchaudio.transforms.resample for caching. [if all orig_sr are the same]
    if kaisier_best:
        return torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=new_sr,lowpass_filter_width=64, \
                                            rolloff=0.9475937167399596,resampling_method="sinc_interp_kaiser",beta=14.769656459379492)
    return torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=new_sr,lowpass_filter_width=32)
        
'''


'''
def load_audio(filepath:str):
    waveform, sr = torchaudio.load(filepath)
    waveform = stereo_to_mono(waveform,dim=-1)
    waveform = resample(waveform,sr,SR)
    waveform = divide(waveform,n=HOP_LENGTH,cut_firstlast_n=int(SR*0.1))
    return waveform

def load_mp3_files(base_folder:str):
    audio_tensors = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                waveform = load_audio(file_path)
                audio_tensors.append(waveform)
    
    
    return audio_tensors, SR

def create_overlapping_chunks_tensor(sequence: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """
    Create overlapping chunks from a tensor.

    Args:
    - sequence (torch.Tensor): Input tensor of shape (1, sequence_length).
    - chunk_len (int): Length of each chunk.

    Returns:
    - torch.Tensor: Tensor of shape (batches, chunk_len) containing overlapping chunks.
    """
    # Ensure the input is a 2D tensor
    if sequence.dim() != 2 or sequence.size(0) != 1:
        raise ValueError("Input tensor must have shape (1, sequence_length)")

    # Reshape the input sequence to a 1D tensor
    sequence = sequence.flatten()  # (sequence_length,)
    
    # Calculate the number of chunks
    sequence_length = sequence.shape[0]
    step_size = chunk_len // 8  # Overlapping by 7/8
    num_chunks = (sequence_length - chunk_len) // step_size + 1
    
    # Create a tensor to hold the chunks
    chunks = torch.zeros((num_chunks, chunk_len))

    # Extract overlapping chunks
    for i in range(num_chunks):
        start_idx = i * step_size
        chunks[i] = sequence[start_idx:start_idx + chunk_len]
    
    return chunks


'''
 maybe better to use nnAudio.features.STFT for trainable layer.

        
        def mel_spectrogram(
            audio: T,
            padding: int = 0,
            device: torch.device = None,
        ):
        
            if padding > 0:
                audio = F.pad(audio, (0, padding)) # TODO: consider DRNN-like architecture or adding tokens after/before the sequence. or pre/post padding.

            transform = torchaudio.transforms.MelSpectrogram(n_fft=N_FFT,win_length=N_FFT,hop_length=HOP_LENGTH,n_mels=N_MELS,window_fn=torch.hann_window,power=2) # real-valued
            mel_spec = transform(audio)
            
            # turn into db scale (log melspectrogram) <- is it really needed for piano besides the idea of normalization?
            #log_spec = torch.log10(torch.clamp(mel_spec, min=1e-8))
            return mel_spec

        def mel_to_audio(mel_spectrogram:T,storeaudioPath:Optional[str]=None):
            print(mel_spectrogram.shape)
            # Step 1: Convert mel-spectrogram back to linear spectrogram
            mel_to_spec_transform = torchaudio.transforms.InverseMelScale(
                n_mels=N_MELS,
                sample_rate=SR,
                n_stft=N_FFT // 2 + 1,  # n_stft is n_fft/2 + 1 by default
            )
            linear_spectrogram = mel_to_spec_transform(mel_spectrogram)
            print(linear_spectrogram.shape)
            griffin_lim = torchaudio.transforms.GriffinLim(n_fft=N_FFT,hop_length=HOP_LENGTH,win_length=N_FFT,window_fn=torch.hann_window,n_iter=64)
            recovered_waveform = griffin_lim(linear_spectrogram) # to make it [1, seq]
            print(recovered_waveform.shape)
            if exists(storeaudioPath):
                torchaudio.save(storeaudioPath, recovered_waveform, SR)

            return recovered_waveform

'''