import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.prototype.functional import chroma_filterbank

from scipy.signal import get_window
from typing import Optional,Callable



class RevSTFT(nn.Module):
    # from: https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py#L456
    def __init__(self,config):
        super().__init__()
        self.filter_length = config['n_fft']
        self.hop_length = config['hop_length']
        self.win_length = config['win_len']

        self.window = torch.from_numpy(get_window('hann', config['win_len'], fftbins=True).astype(np.float32))

    def transform(self, input_data):
        if input_data.dim() == 3 and input_data.size(1) == 1:
            input_data.squeeze(1) 
        
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform) 
        # mag & phase. phase is being used but maybe tough for the model to learn, but essential for high quality.

    def inverse(self, magnitude, phase):
        
        magnitude = magnitude.transpose(-1,-2)
        phase = phase.transpose(-1,-2)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j), # waveform = mag * e^phase
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform
    
    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)

        self.magnitude = self.magnitude.transpose(-1,-2)
        self.phase = self.phase.transpose(-1,-2)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
    
    @property
    def frequencies(self):
        """
        Get frequency bins in Hz
        
        Returns:
            torch.Tensor: Frequency bins
        """
        return torch.fft.rfftfreq(self.filter_length) * self.filter_length



# from official torchaudio documentation. Had to get the source code as only nightly version of torchaudio supports those modules.
class ChromaScale(nn.Module):
    r"""Converts spectrogram to chromagram.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_freqs (int): Number of frequency bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_chroma (int, optional): Number of chroma. (Default: ``12``)
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> spectrogram_transform = transforms.Spectrogram(n_fft=1024)
        >>> spectrogram = spectrogram_transform(waveform)
        >>> chroma_transform = transforms.ChromaScale(sample_rate=sample_rate, n_freqs=1024 // 2 + 1)
        >>> chroma_spectrogram = chroma_transform(spectrogram)

    See also:
        :py:func:`torchaudio.prototype.functional.chroma_filterbank` — function used to
        generate the filter bank.
    """

    def __init__(
        self,
        sample_rate: int,
        n_freqs: int,
        *,
        n_chroma: int = 12,
        tuning: float = 0.0,
        ctroct: float = 5.0,
        octwidth: Optional[float] = 2.0,
        norm: int = 2,
        base_c: bool = True,
    ):
        super().__init__()
        fb = chroma_filterbank(
            sample_rate, n_freqs, n_chroma, tuning=tuning, ctroct=ctroct, octwidth=octwidth, norm=norm, base_c=base_c
        )
        self.register_buffer("fb", fb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            specgram (torch.Tensor): Spectrogram of dimension (..., ``n_freqs``, time).

        Returns:
            torch.Tensor: Chroma spectrogram of size (..., ``n_chroma``, time).
        """
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)



class ChromaSpectrogram(nn.Module):
    r"""Generates chromagram for audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Composes :py:func:`torchaudio.transforms.Spectrogram` and
    and :py:func:`torchaudio.prototype.transforms.ChromaScale`.

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., torch.Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        n_chroma (int, optional): Number of chroma. (Default: ``12``)
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.ChromaSpectrogram(sample_rate=sample_rate, n_fft=400)
        >>> chromagram = transform(waveform)  # (channel, n_chroma, time)
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        *,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        n_chroma: int = 12,
        tuning: float = 0.0,
        ctroct: float = 5.0,
        octwidth: Optional[float] = 2.0,
        norm: int = 2,
        base_c: bool = True,
    ):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        #from torchaudio.prototype.functional import chroma_filterbank

        self.chroma_scale = ChromaScale(
            sample_rate,
            n_fft // 2 + 1,
            n_chroma=n_chroma,
            tuning=tuning,
            base_c=base_c,
            ctroct=ctroct,
            octwidth=octwidth,
            norm=norm,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Chromagram of size (..., ``n_chroma``, time).
        """
        spectrogram = self.spectrogram(waveform)
        chroma_spectrogram = self.chroma_scale(spectrogram)
        return chroma_spectrogram

