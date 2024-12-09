import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
import math
# Karras et al. https://arxiv.org/pdf/2206.00364 implementation
from tqdm import trange
from layers.cores.dit import DiT
    # Somewhat not working
class Denoiser(nn.Module):
    # mostly from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
    # also from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/inference/sampling.py#L179
    def __init__(
        self,
        config,
        model: nn.Module = DiT(),
        sigma_data: float = 1,  
        sigma_min : float = 1e-2,
        sigma_max : float = 5.0,
        rho: float = 7.0, 
        s_churn: float = 0.0, 
        s_tmin: float = 0.05, 
        s_tmax: float = 1e+8, 
        s_noise: float = 1.001, 
        device: torch.device = torch.device("cuda:0")
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max 
        self.rho = rho
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

        self.rng = torch.quasirandom.SobolEngine(1, scramble = True, seed = 42)

    def sigma_noise(self, num_samples, stratified = True):
        if stratified:  # somewhat V-diffusion uses this
            return (self.rng.draw(num_samples) * 1.2 - 1.2).exp().to(self.device)
        return (torch.rand((num_samples), device = self.device)[:, None] * 1.2 - 1.2).exp()

    def get_scalings(self, sigmas):
        c_skip = self.sigma_data ** 2 / (sigmas ** 2 + self.sigma_data ** 2)
        c_out = sigmas * self.sigma_data / (sigmas ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigmas ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = sigmas.log() / 4
        return c_skip, c_out, c_in, c_noise
    
    def append_dims(self, x, target_dims):
        if target_dims < x.ndim:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x[(...,) + (None,) * (target_dims - x.ndim)]


    def forward(self, x : Tensor, sigmas = None) -> Tensor:
        
        x = self.model.autoencoder.encode_audio(x)  # audio -> latent
        x_std = x.std(dim=1,keepdim=True)
        x /= x_std
        sigmas = self.sigma_noise(x.shape[0]) if sigmas is None else sigmas
        
        c_skip, c_out, c_in, c_noise = [self.append_dims(cond, x.ndim) for cond in self.get_scalings(sigmas)]
        x_denoised = self.model.forward_latent(c_in *  x, c_noise) * c_out + c_skip * x
        
        #x_denoised = self.model.autoencoder.decode_audio(x_denoised)
        return x_denoised * x_std, sigmas

    def loss_fn(self, x : Tensor, sigmas = None):
        x = self.model.autoencoder.encode_audio(x)  # audio -> latent
        x_std_chn = x.std(dim=1, keepdim=True)
        x /= x_std_chn
        sigmas = self.sigma_noise(x.shape[0]) if sigmas is None else sigmas


        noise = torch.randn_like(x) 
        noised_x = x + noise * sigmas
        c_skip, c_out, c_in, c_noise = [self.append_dims(cond, x.ndim) for cond in self.get_scalings(sigmas)]
        x_denoised = self.model.forward_latent(c_in * noised_x, c_noise) 
        target = (x - c_skip * noised_x) / c_out 
        
        loss = (x_denoised - target).pow(2) 
        loss *= self.sigma_data ** 2 / (sigmas ** 2 + self.sigma_data ** 2) # snr weightning is known to be better than karras's one.
        loss = loss.reshape(-1).mean()
        return loss

    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
    ) -> Tensor:
        
        sigmas = self._schudule_sigmas(100)
        sigmas = sigmas[:, None, None]
        x = self.model.autoencoder.encode_audio(torch.randn((num_samples,self.config['seq_len']),device=self.device) )
        x = torch.randn_like(x) * sigmas[0]
        generated_latent = self.sample_heun(x, sigmas)
        x = self.model.autoencoder.decode_audio(generated_latent)

        return x
        
        

    @torch.no_grad()
    def _schudule_sigmas(self, num_steps: int):
        dm = self.sigma_max ** (1.0 / self.rho)
        mm = self.sigma_min ** (1.0 / self.rho)
        s = (dm + torch.linspace(0, 1, num_steps - 1, device = self.device) * (mm - dm)) ** self.rho
        return torch.cat((s, s.new_zeros([1])))
    
    @torch.no_grad()
    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    @torch.no_grad()
    def sample_heun(self, x, sigmas, s_churn=40., s_tmin=0., s_tmax=float('inf'), s_noise=1.001):
        """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            denoised = self.model.forward_latent(x, sigma_hat * s_in)
            d = self.to_d(x, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.model.forward_latent(x_2, sigmas[i + 1] * s_in)
                d_2 = self.to_d(x_2, sigmas[i + 1], denoised_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x
        

    

