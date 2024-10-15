import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
# config
SR = 8000
N_FFT = 512
WIN_LEN = 512
HOP_LENGTH = 256
CHUNK_LENGTH = 30
N_MELS = 64
MIN=1e-7
MAX=2e+5

def stft(x, fft_size, hop_size, win_size, window):
    # do stft and gets magnitude
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window,return_complex=True)
    x_real = torch.view_as_real(x_stft)
    x_real = torch.clamp(x_real[...,0]**2 + x_real[...,1]**2,min=MIN)

    outputs = torch.sqrt(x_real).transpose(2, 1) # clamp first


    return outputs 

class SpectralConvergence(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predicts_mag, targets_mag):
        
        return torch.norm(targets_mag - predicts_mag, p=1) / torch.norm(targets_mag, p=1)

class LogSTFTMagnitude(nn.Module):
    def __init__(self):
        super(LogSTFTMagnitude, self).__init__()
        
    def forward(self, predicts_mag, targets_mag):
        log_predicts_mag = torch.log(predicts_mag)
        log_targets_mag = torch.log(targets_mag)
        outputs = F.l1_loss(log_predicts_mag, log_targets_mag)

        return outputs

class STFTLoss(nn.Module):
    def __init__(self,
                 fft_size=N_FFT,
                 hop_size=HOP_LENGTH,
                 win_size=WIN_LEN):
        super().__init__()
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = torch.hann_window(win_size)
        self.sc_loss = SpectralConvergence()
        self.mag = LogSTFTMagnitude()
        
    
    def forward(self, predicts, targets):
 
        predicts_mag = stft(predicts, self.fft_size, self.hop_size, self.win_size, self.window)
        targets_mag = stft(targets, self.fft_size, self.hop_size, self.win_size, self.window)
        
        sc_loss = self.sc_loss(predicts_mag, targets_mag)
        mag_loss = self.mag(predicts_mag, targets_mag)
        print(f"sc: {sc_loss.item()} mag: {mag_loss.item()}")
        return sc_loss,mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes=[N_FFT, N_FFT*2, N_FFT*4],
                 win_sizes=[WIN_LEN, WIN_LEN*2, WIN_LEN*4],
                 hop_sizes=[HOP_LENGTH, WIN_LEN*2, HOP_LENGTH*4]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size))
            
    def forward(self, fake_signals, true_signals):
        sc_losses, mag_losses = [], []
        for laye3r in self.loss_layers:
            sc_loss, mag_loss = laye3r(fake_signals, true_signals)
            sc_losses.append(sc_loss)
            mag_losses.append(mag_loss)
        
        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)

        return 0.5*sc_loss + 0.5*mag_loss


class DynamicLoss(nn.Module):
    
    """
        taking loss on the original sequences. expect [B,1,L] shapes of x,y.

        CTC:  Calculates loss between a continuous (unsegmented) time series and a target sequence
        CosineSIM: Loss based on A*B/(||A|| * ||B||). inspired by https://arxiv.org/pdf/2212.07669's usage. 
                    meaning while the sequence are not idetical, they can still a generate decent audio. (or facing the right direction)

        PerceptronLoss: For helping converging fast at beginning. https://see.stanford.edu/materials/aimlcs229/cs229-notes6.pdf

    """
    def __init__(self):
        super().__init__()
        
        self.mse =  nn.MSELoss()#HingeEmbeddingLoss(margin=0)#CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.cosineEmbed = nn.CosineEmbeddingLoss(margin=0)
        self.stftLoss = MultiResolutionSTFTLoss()
        self.lambda_val = 0.7
        self.update_rate = 0.01 # for every epoch
        self.repeat_cycle = 5

    def forward(self, x, y):
        assert x.size() == y.size()
        return self.mse(x,y)
        # Calculate CTC loss
        ctc_loss = self.mse(x, y)
        print(f"CTC Loss: {ctc_loss.item()}")

        # Calculate Cosine Embedding loss
        cosineEmbed = self.cosineEmbed(x, y, torch.ones(x.size(0)))
        print(f"Cosine Embedding Loss: {cosineEmbed.item()}")

        # Calculate STFT loss
        stft_loss = self.stftLoss(x, y)
        print(f"STFT Loss: {stft_loss.item()}")


        # Combine the losses with lambda weighting
        total_loss = (1 - self.lambda_val) * (0.5*ctc_loss + 0.5*stft_loss) + self.lambda_val * cosineEmbed 
        print(f"Total Loss: {total_loss.item()}")

        return total_loss

    def perceptronLoss(self,x,y):
        return torch.clamp(-x * y, min=0).mean()
    
    def lambda_update(self,):
        if (self.repeat_cycle == 0):
            self.lambda_val = 0.05
            return

        self.lambda_val -= self.update_rate
        if (self.lambda_val < 0.01): # once reached too small -> turn into linear cycle mode.
            self.lambda_val = 0.2
            self.repeat_cycle -= 1
        
    

