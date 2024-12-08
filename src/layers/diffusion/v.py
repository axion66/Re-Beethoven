import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
import math
# Karras et al. https://arxiv.org/pdf/2206.00364 implementation
from tqdm import trange

class Denoiser(nn.Module):
    # mostly from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
    # also from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/inference/sampling.py#L179
    def __init__(
        self,
        config,
        model: nn.Module,
        sigma_data: float = 0.5,  
        sigma_min : float = 0.002,
        sigma_max : float = 1.0,
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

        self.mse = nn.MSELoss()
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True) 

    def forward(self, x: Tensor) -> Tensor:
        t = self.rng.draw(x.shape[0])[:, 0].to(self.device)
        alphas, sigmas = self.get_alphas_sigmas(t)
        alphas = alphas[:, None]
        sigmas = sigmas[:, None]
        noise = torch.randn_like(x)
        noised_inputs = x * alphas + noise * sigmas
        out = self.model(noised_inputs, t.unsqueeze(-1))

        return x * alphas - out * sigmas, t      
        

    def get_alphas_sigmas(self, t):

        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


    def loss_fn(self, x:Tensor):
        t = self.rng.draw(x.shape[0])[:, 0].to(self.device) 
        alphas, sigmas = self.get_alphas_sigmas(t)
        alphas = alphas[:, None]
        sigmas = sigmas[:, None]
        noise = torch.randn_like(x)
        noised_inputs = x * alphas + noise * sigmas
        targets = noise * alphas - x * sigmas
        out = self.model(noised_inputs, t)
        return self.mse(out, targets)
     

    @torch.no_grad()
    def sample(self, num_samples):
        """Draws samples from a model given starting noise. v-diffusion"""
        x = torch.randn((num_samples, self.config['seq_len']), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        steps = 100
        eta = 0  
        ts = x.new_ones([x.shape[0]])
        t = torch.linspace(1, 0, steps + 1)[:-1]

        alphas, sigmas = self.get_alphas_sigmas(t)
        # The sampling loop
        for i in trange(steps):

            # Get the model output (v, the predicted velocity)
            with torch.amp.autocast(device_type = "cuda" if torch.cuda.is_available() else "cpu"):
                v = self.model(x, ts * t[i]).float()

            # Predict the noise and the denoised image
            pred = x * alphas[i] - v * sigmas[i]
            eps = x * sigmas[i] + v * alphas[i]

            # If we are not on the last timestep, compute the noisy image for the
            # next timestep.
            if i < steps - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                    (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                x = pred * alphas[i + 1] + eps * adjusted_sigma

                # Add the correct amount of fresh noise
                if eta:
                    x += torch.randn_like(x) * ddim_sigma

    
        # If we are on the last timestep, output the denoised image
        return pred

    
