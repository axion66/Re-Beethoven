import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)



class PerceptronLoss(nn.Module):
    # we only want pos/neg change. No need for hinge loss.
    def __init__(self):
        super(PerceptronLoss, self).__init__()
    
    def forward(self, outputs, targets):
        return torch.clamp(-outputs * targets, min=0).mean()
         



class CTCCosinePerceptronLoss(nn.Module):
    
    """
        CTC: 
    
    
    """
    def __init__(self):
        super().__init__()
        
        self.ctc =  nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.cosineEmbed = nn.CosineEmbeddingLoss(margin=0)
        self.perceptron = PerceptronLoss()
    def forward(self,x,y):
        assert x.size() == y.size()
        input_lengths = torch.full((x.size(0),), x.size(-1), dtype=torch.long)  
        target_lengths = torch.full((y.size(0),), y.size(-1), dtype=torch.long)
        return 0.6 * self.ctc(x,y,input_lengths,target_lengths) + \
            0.3 * self.cosineEmbed(x,y,torch.ones(x.size(0))) + 0.2 * self.perceptron(x,y)
    
    
class SpectralConvergengeLoss(torch.nn.Module):
    # https://github.com/rishikksh20/UnivNet-pytorch/blob/master/stft_loss.py
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window.to(x.get_device()))
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(x.get_device()))
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss
    



class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(GeneratorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)


        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss.mean()
   
    
class WavLMLoss(torch.nn.Module):

    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
     
    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16.squeeze(), output_hidden_states=True).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))
        
        return floss.mean()
    
    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen
    
    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        
        return y_d_rs