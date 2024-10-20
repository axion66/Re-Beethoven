import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from typing import Callable

# All of them are from Karras et al. paper: https://arxiv.org/pdf/2206.00364
# TODO: do test w/ small dSEt & hyperparameter tuning 

def exists(val):
    return val is not None




class NoiseDistribution:
    # Noise distribution (Section 5)
    def __init__(self, mean: float = -1.2, std: float = 1.2):
        self.mean = mean
        self.std = std

    def __call__(
        self, 
        num_samples: int, 
        device: torch.device = torch.device("cuda:0")
    ) -> Tensor:
        
        return torch.normal(self.mean,self.std,size=(num_samples,),device=device).exp()


class Denoiser(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        sigma_data: float=0.5,  # data distribution standard deviation
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.get_noise = NoiseDistribution() # sigma

    def get_scaling(self, sigmas: Tensor):
        # network and preconditioning
        sigmas = rearrange(sigmas, "b -> b 1")
        c_common = sigmas**2 + self.sigma_data**2 # sig**2 + sig_data**2
        c_skip = (self.sigma_data ** 2) / c_common # c_skip
        c_out = sigmas * self.sigma_data / (c_common ** 0.5) # c_out
        c_in = 1 / (c_common ** 0.5) # c_in
        c_noise = torch.log(sigmas) * 0.25 # c_noise
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Tensor,
        **kwargs,
    ) -> Tensor:

        c_skip, c_out, c_in, _ = self.get_scaling(sigmas)
        x_pred = self.model(c_in * x_noisy, sigmas)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        return x_denoised

    def loss_weighting(self, sigmas: Tensor) -> Tensor:
        # lambda(sigma)
        return (sigmas ** 2 + self.sigma_data ** 2) / (sigmas * self.sigma_data) ** 2

    def forward(self, x: Tensor) -> Tensor:
        # TODO: seperate calculating loss and forward method
        b, device = x.shape[0], x.device
        sigmas = self.get_noise(num_samples=b, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")
        noise = torch.randn_like(x)
        x_noisy = x + sigmas_padded.squeeze(1) * noise
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas)
        
        return x_denoised,sigmas

    def calculate_loss(self,x:Tensor,x_denoised:Tensor,sigmas:Tensor):
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses *= self.loss_weighting(sigmas)
        loss = losses.mean() 
        return loss



class KarrasSampler(nn.Module):
    # Heun's 2nd order method (ODE solver) more accurate, but more cost


    def __init__(
        self,
        s_churn: float = 40.0, # controls stochasticity  0 for deterministic
        s_tmin: float = 0.05, # I need to find with grid search, but who wants to do that..
        s_tmax: float = 999999, # Figure 15 (yellow line)
        s_noise: float = 1.003 # to inflate std for newly added noise.
        
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_churn = s_churn
        self.s_noise = s_noise

    @torch.no_grad()
    def denoise(
        self, x: Tensor, model: Callable, sigma: float, sigma_next: float, gamma: float
    ) -> Tensor:

        
        epsilon = (self.s_noise**2) * torch.randn_like(x) # sample e_i ~ N(0,S_noise^2)
        sigma_hat = sigma * (gamma + 1) # sigma_hat <- sigma_i + gamma*sigma_i
        x_hat = x + ((sigma_hat) ** 2 - sigma ** 2)**0.5 * epsilon # x_hat <- x + sqrt(sigma_hat^2 - sigma_i^2) * eps
        d = (x_hat - model(x_hat, sigma=sigma_hat)) / sigma_hat # d = (x_hat - Denoise_func(x_hat, sigma_hat)) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d # update x with estimated noise. d has a denominator of sigma_hat.
        if sigma_next != 0: # if not final layer, then apply 2nd order correction (revise x_next using sigma_next)
            model_out_next = model(x_next, sigma=sigma_next) # calculate x_next
            d_prime = (x_next - model_out_next) / sigma_next # subtract noise
            x_next = x_hat + (sigma_next - sigma_hat) * 0.5 * (d + d_prime) # 2nd order correction (weight: d = 0.5, d_prime = 0.5)
        return x_next

    
    def forward(
        self, noise: Tensor, model: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = (sigmas[0]**2) * noise # sample x_0 ~ N(0,sigmas[0])
        gammas = torch.where( # gamma for higher noise level 
            (self.s_tmin <= sigmas <= self.s_tmax),
            min(self.s_churn / num_steps, 0.41421356237309504), # 0.4142 ~ sqrt(2) - 1
            0.0,
        )

        for i in range(num_steps - 1):
            x = self.denoise(
                x, model=model, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i]  # type: ignore # noqa
            )

        return x




class KarrasSchedule(nn.Module):
    # Constructs the noise schedule (for Karras Diffusion)
    def __init__(self, sigma_data=0.5,sigma_min=0.002, sigma_max=80, rho: float = 7.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.rho_inverse = 1.0 / rho
        
    def forward(self, num_steps: int, device: torch.device = torch.device("cuda:0")) -> Tensor:
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** self.rho_inverse
            + (steps / (num_steps - 1)) * (self.sigma_min ** self.rho_inverse - self.sigma_max ** self.rho_inverse)
        ) ** self.rho
        sigmas = torch.cat((sigmas,sigmas.new_zeros([1])))
        return sigmas


