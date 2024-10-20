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
    ) -> Tensor:
        
        c_skip, c_out, c_in, _ = self.get_scaling(sigmas)
        x_pred = self.model(c_in * x_noisy, sigmas)

        x_denoised = c_skip * x_noisy + c_out * x_pred

        return x_denoised

    def loss_weighting(self, sigmas: Tensor) -> Tensor:
        # lambda(sigma)
        return (sigmas ** 2 + self.sigma_data ** 2) / (sigmas * self.sigma_data) ** 2

    def forward(self, x: Tensor, sigma=None) -> Tensor:
        # TODO: seperate calculating loss and forward method
        b, device = x.shape[0], x.device
        sigmas = self.get_noise(num_samples=b, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")
        noise = torch.randn_like(x)
        x_noisy = x + sigmas_padded.squeeze(1) * noise
        sigmas = sigma if exists(sigma) else sigmas
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
        self, x: Tensor, model: Callable, sigma: Tensor, sigma_next: Tensor, gamma: Tensor
    ) -> Tensor:
        epsilon = (self.s_noise**2) * torch.randn_like(x)
        sigma_hat = sigma * (gamma + 1)
        x_hat = x + ((sigma_hat) ** 2 - sigma ** 2)**0.5 * epsilon
        
        # Create a sigma_hat tensor of the same shape as x
        sigma_hat_expanded = sigma_hat.expand(x.shape[0], 1)
        d = (x_hat - model.denoise_fn(x_hat, sigma_hat_expanded.squeeze(-1))) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat.squeeze(-1)) * d
        
        if not torch.all(sigma_next == 0):  # Check if any sigma_next is non-zero
            # Create a sigma_next tensor of the same shape as x
            sigma_next_expanded = sigma_next.expand(x.shape[0], 1)
            
            model_out_next = model.denoise_fn(x_next, sigma_next_expanded.squeeze(-1))
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * 0.5 * (d + d_prime)
        
        return x_next

    def forward(
        self, noise: Tensor, model: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0].pow(2) * noise  # Use .pow(2) instead of **2 for better compatibility
        
        gammas = torch.where(
            (self.s_tmin <= sigmas) & (sigmas <= self.s_tmax),
            torch.tensor(min(self.s_churn / num_steps, 0.41421356237309504)),
            torch.tensor(0.0)
        )
        
        for i in range(num_steps - 1):
            x = self.denoise(
                x, 
                model=model, 
                sigma=sigmas[i].unsqueeze(0),  # Add batch dimension
                sigma_next=sigmas[i + 1].unsqueeze(0),  # Add batch dimension
                gamma=gammas[i].unsqueeze(0)  # Add batch dimension
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


