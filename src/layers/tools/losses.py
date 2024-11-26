
import torch
import torch.nn as nn
from typing import List, Any

# https://github.com/csteinmetz1/auraloss/tree/main
# paper suggests L1 Loss & MR is the best performing one..

class Loss(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        
    
class LossModule:
    def __init__(self, name):
        self.name = name
        self.losses : list = []
        self.discriminator_step = 0
        
    def append(self, name, weight_loss, module: Loss):
        self.losses.append(
            {
                "name": name,
                "weight": weight_loss,
                "module": module,
            }
            
        )
        
    def loss_fn(self, x, y):

        total_loss = 0.0
        info = {}
        
        for loss_module in self.losses:
            
            name = loss_module['name']
            weight = loss_module['weight']
            
            l = loss_module['module'](x, y)
            l *= weight
            total_loss += l
            info.update({name : l})
            '''
            if name.startswith("Discri") or name.startswith("discri"): # meaning discriminator
                self.discriminator_step += 1
                if (self.discriminator_step%2 == 0 and self.discriminator_step < 20000):  # then discriminator
                    l = loss_module['module'].loss(x.unsqueeze(1), y.unsqueeze(1))
                    l_total = l[0] + l[1] +  l[2]
                    l_total *= weight
                    info = {}
                    info.update({
                        f"{name}/dis_loss":l[0] * weight,
                        f"{name}/adv_loss": l[1] * weight,
                        f"{name}/feature_matching_loss": l[2] * weight,
                    })
                    return l_total, info
                
                
            else:
                l = loss_module['module'](x, y)
                l *= weight
                total_loss += l
                info.update({name : l})
            '''
        info.update({"total loss": total_loss})
        
        return total_loss, info
    
class L1Loss(Loss):
    def __init__(self,distance='mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=distance)
    def forward(self,x,y):
        return self.loss(x, y)

class SpectralConvergenceLoss(Loss):
    """Spectral convergence loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        return (torch.norm(y_mag - x_mag, p="fro", dim=[-1, -2]) / torch.norm(y_mag, p="fro", dim=[-1, -2])).mean()

class STFTMagnitudeLoss(Loss):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 0.0
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()

        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)

class STFTLoss(Loss):
    """STFT loss module.

    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).

    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        mag_distance (str, optional): Distance function ["L1", "L2"] for the magnitude loss terms.
        device (str, optional): Place the filterbanks on specified device. Default: None

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'. By default.
        loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = "chroma",
        n_bins: int = None,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        **kwargs
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device
        self.phs_used = bool(self.w_phs)

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction,
            distance=mag_distance,
            **kwargs
        )
        self.linstft = STFTMagnitudeLoss(
            log=False,
            reduction=reduction,
            distance=mag_distance,
            **kwargs
        )

        # setup mel filterbank
        if scale is not None:
            try:
                import librosa.filters
            except Exception as e:
                print(e)
                print("Try `pip install auraloss[all]`.")

            if self.scale == "mel":
                assert sample_rate != None  # Must set sample rate to use mel scale
                assert 32 <= fft_size  # Must be more FFT bins than Mel bins
                fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=32)
                fb = torch.tensor(fb).unsqueeze(0)

            elif self.scale == "chroma":
                assert sample_rate != None  # Must set sample rate to use chroma scale
                assert 32 <= fft_size  # Must be more FFT bins than chroma bins
                fb = librosa.filters.chroma(
                    sr=sample_rate, n_fft=fft_size, n_chroma=32
                )
                torch.tensor(fb).unsqueeze(0)

            else:
                raise ValueError(
                    f"Invalid scale: {self.scale}. Must be 'mel' or 'chroma'."
                )

       
            self.register_buffer("fb", torch.tensor(fb))

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

    
    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps)
        )

        # torch.angle is expensive, so it is only evaluated if the values are used in the loss
        if self.phs_used:
            x_phs = torch.angle(x_stft)
        else:
            x_phs = None

        return x_mag, x_phs

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(input.device)

        x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
        y_mag, y_phs = self.stft(target.view(-1, target.size(-1)))

        # apply relevant transforms
        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = torch.nn.functional.mse_loss(x_phs, y_phs) if self.phs_used else 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
        )

        loss = loss.mean()

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss

class MultiResolutionSTFTLoss(Loss):
    """Multi resolution STFT loss module.

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: 'chroma'
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """
    '''
    "fft_sizes": [2048, 1024, 512, 256, 128, 64, 32],
                    "hop_sizes": [512, 256, 128, 64, 32, 16, 8],
                    "win_lengths": [2048, 1024, 512, 256, 128, 64, 32],
                    "perceptual_weighting": true
    '''

    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512, 256, 128, 64, 32],
        hop_sizes: List[int] = [512, 256, 128, 64, 32, 16, 8],
        win_lengths: List[int] = [2048, 1024, 512, 256, 128, 64, 32],
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        scale_invariance: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    w_sc,
                    w_log_mag,
                    w_lin_mag,
                    w_phs,
                    sample_rate,
                    scale,
                    n_bins,
                    scale_invariance,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        #https://static1.squarespace.com/static/5554d97de4b0ee3b50a3ad52/t/5fb1e9031c7089551a30c2e4/1605495044128/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for f in self.stft_losses:
            if f.output == "full":  # extract just first term
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)

        mrstft_loss /= len(self.stft_losses)

        if f.output == "loss":
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss




def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
    return dis_loss, gen_loss

class EncodecDiscriminator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        from encodec.msstftd import MultiScaleSTFTDiscriminator

        self.discriminators = MultiScaleSTFTDiscriminator(filters=64)

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, y, x):
        feature_matching_distance = 0.
        logits_true, feature_true = self.forward(x)
        logits_fake, feature_fake = self.forward(y)

        dis_loss = torch.tensor(0.)
        adv_loss = torch.tensor(0.)

        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = get_hinge_losses(
                logits_true[i],
                logits_fake[i],
            )

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return dis_loss, adv_loss, feature_matching_distance



'''
class EncodecDiscriminator(Loss):
    def __init__(self):
        super().__init__()

        from encodec.msstftd import MultiScaleSTFTDiscriminator
        self.discriminators = MultiScaleSTFTDiscriminator(filters=64,
                                                          n_ffts=[2048, 1024, 512, 256, 128],
                                                          hop_lengths=[512, 256, 128, 64, 32],
                                                          win_lengths=[2048, 1024, 512, 256, 128],
                                                          )

    def _forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def get_hinge_losses(self, score_real, score_fake):
        gen_loss = -score_fake.mean() # why not (1 - score_Fake)?
        dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
        return dis_loss, gen_loss
    
    def forward(self, y, x): # fake for y, true for x
        feature_matching_distance = 0.
        logits_true, feature_true = self._forward(x)
        logits_fake, feature_fake = self._forward(y)

        dis_loss = torch.tensor(0.)
        adv_loss = torch.tensor(0.)

        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = self.get_hinge_losses(
                logits_true[i],
                logits_fake[i],
            )

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return 1 * dis_loss, 0.1 * adv_loss, 5 * feature_matching_distance
        # use dis_loss only for discriminator, and use adv_loss and feature_matching_distance for generator.
'''