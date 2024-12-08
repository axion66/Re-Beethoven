import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
import math
# Karras et al. https://arxiv.org/pdf/2206.00364 implementation
from tqdm import trange

    # Somewhat not working
class Denoiser(nn.Module):
    # mostly from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
    # also from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/inference/sampling.py#L179
    def __init__(
        self,
        config,
        model: nn.Module,
        sigma_data: float = 0.5,  
        sigma_min : float = 1e-2,
        sigma_max : float = 80.0,
        rho: float = 7.0, 
        s_churn: float = 40.0, 
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
        # self.sigma_noise = lambda num_samples: (torch.rand((num_samples, 1), device = device) * 1.2 - 1.2).exp()
        # can't believe I used torch.randn instead of torch.rand here..

        if stratified:
            return (self.rng.draw(num_samples, device = self.device)[:, None] * 1.2 - 1.2).exp()
        return (torch.rand(num_samples, device = self.device)[:, None] * 1.2 - 1.2).exp()

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
        
        if (sigmas is None):
            sigmas = self.sigma_noise(x.shape[0])
        
        noise = torch.randn_like(x) * sigmas
        noised_x = x + noise
        c_skip, c_out, c_in, c_noise = [self.append_dims(cond, x.ndim) for cond in self.get_scalings(sigmas)]
        x_denoised = self.model(c_in *  noised_x, c_noise) * c_out + noised_x * c_skip
        
        return x_denoised, sigmas

    def loss_fn(self, x : Tensor, sigmas = None):
 
        if sigmas is None:
            sigmas = self.sigma_noise(x.shape[0])

        noise = torch.randn_like(x) * sigmas
        noised_x = x + noise
        c_skip, c_out, c_in, c_noise = [self.append_dims(x, x.ndim) for x in self.get_scalings(sigmas)]
        x_denoised = self.model(c_in * noised_x, c_noise) 
        target = (x - c_skip * noised_x) / c_out 
        
        loss = ((x_denoised - target)**2)
        snr_weight = self.sigma_data ** 2 / (sigmas ** 2 + self.sigma_data ** 2)
        loss *= snr_weight
        return loss.reshape(-1).mean()

    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
    ) -> Tensor:
        

        sigmas = self._schudule_sigmas(100)[:, None]

        x = sigmas[0] ** 2 * torch.randn((num_samples,self.config['seq_len']),device=self.device) 
        gammas = torch.where(
            (self.s_tmin <= sigmas) & (sigmas <= self.s_tmax),
            torch.tensor(min(self.s_churn / 100, math.sqrt(2) - 1)), 
            torch.tensor(0.0)
        )

        for i in range(100 - 1):
            x = self._heun_method(
                x, 
                sigma=sigmas[i], 
                sigma_next=sigmas[i + 1], 
                gamma=gammas[i]
            )
        
        return x
    

    @torch.no_grad()
    def _schudule_sigmas(self, num_steps: int):
        dm = self.sigma_max ** (1.0 / self.rho)
        mm = self.sigma_min ** (1.0 / self.rho)
        s = (dm + torch.linspace(0, 1, num_steps - 1, device = self.device) * (mm - dm)) ** self.rho
        return torch.cat((s, s.new_zeros([1])))
    

    @torch.no_grad()
    def _heun_method(
        self,
        x: Tensor,
        sigma: Tensor,
        sigma_next: Tensor,
        gamma: Tensor
    ) -> Tensor:
        
        epsilon = (self.s_noise ** 2) * torch.randn_like(x)
        sigma_hat = sigma * (1 + gamma)
        delta_sigma = sigma_next - sigma_hat
        if (gamma.values > 0):
            x = x + (sigma_hat ** 2 - sigma ** 2)**0.5 * epsilon

        x_first = self.model(x, sigma_hat[:, None])
        d_first = (x - x_first) / sigma_hat
        
        x_next = x + delta_sigma * d_first
        # Heun's second difference
        if sigma_next.values != 0.0:
            x_second = self.model(x_next, sigma_next[:, None])  
            d_second = (x_next - x_second) / sigma_next

            d_prime = (d_first + d_second) / 2
            x_next = x + delta_sigma * d_prime
        
        return x_next
    
