import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

# Karras et al. https://arxiv.org/pdf/2206.00364 implementation



class Denoiser(nn.Module):

    def __init__(
        self,
        config,
        model: nn.Module,
        sigma_data: float = 0.5,  
        sigma_min : float = 0.002,
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
        self.sigma_noise = lambda num_samples: (torch.randn((num_samples, 1), device = device) - 0.4).exp()
        

    def get_scalings(self,sigmas):
        c_skip = (self.sigma_data ** 2) / (sigmas**2 + self.sigma_data**2)
        c_out = sigmas * self.sigma_data / ((sigmas**2 + self.sigma_data**2) ** 0.5) 
        c_in = 1 / ((sigmas**2 + self.sigma_data**2) ** 0.5) 
        c_noise = sigmas.log() / 4 
        return c_skip,c_out,c_in,c_noise


    def append_dims(self,x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x[(...,) + (None,) * dims_to_append]


    def forward(self, x: Tensor, sigmas = None) -> Tensor:
        
        sigmas = self.sigma_noise(num_samples = x.shape[0]) if sigmas is None else sigmas
        while (sigmas.ndim < x.ndim):
            sigmas = sigmas.unsqueeze(-1)

        noise = torch.randn_like(x) * sigmas
        c_skip, c_out, c_in, c_noise = [self.append_dims(x, x.ndim) for x in self.get_scalings(sigmas)]
        x_denoised = self.model(c_in * (x + noise), c_noise) * c_out + x * c_skip
        
        return x_denoised, sigmas

    def loss_fn(self, x:Tensor):
 

        sigmas = self.sigma_noise(num_samples = x.shape[0])
        while (sigmas.ndim < x.ndim):
            sigmas = sigmas.unsqueeze(-1)
        x_noised = x + (torch.randn_like(x) * sigmas) # randn_like * sigmas == noise


        c_skip, c_out, c_in, c_noise = [self.append_dims(x, x.ndim) for x in self.get_scalings(sigmas)]
        x_denoised = self.model(c_in * x_noised, c_noise) #   * c_out + x * c_skip -> replaced by changing original x.
        x = (x - c_skip * x_noised) / c_out 
        
        snr_weight = self.sigma_data ** 2 / (sigmas ** 2 + self.sigma_data ** 2)
        loss = snr_weight * ((x_denoised - x)**2)
        
        return loss.reshape(-1).mean()


    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int,
    ) -> Tensor:
        

        sigmas = self._schudule_sigmas(num_steps).unsqueeze(-1) # t = {batch,1}

        x = sigmas[0] ** 2 * torch.randn((num_samples,self.config['seq_len']),device=self.device) 
        gammas = torch.where(
            (self.s_tmin <= sigmas) & (sigmas <= self.s_tmax),
            torch.tensor(min(self.s_churn/num_steps, 0.414213)), # 0.4142 ~ sqrt(2) - 1
            torch.tensor(0.0)
        )

        for i in range(num_steps - 1):
            x = self._heun_method(
                x, 
                sigma=sigmas[i], #t_i
                sigma_next=sigmas[i + 1], # t_(i+1)
                gamma=gammas[i]
            )
        
        return x
    
    @torch.no_grad()
    def _schudule_sigmas(self, num_steps: int):
        steps = torch.arange(num_steps, device=self.device, dtype=torch.float32) 
        schuduled_sigmas = (
            self.sigma_max ** (1.0 / self.rho)
            + (steps / (num_steps - 1)) * (self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho))
        ) ** self.rho

        schuduled_sigmas = torch.cat((schuduled_sigmas,schuduled_sigmas.new_zeros([1])))
        return schuduled_sigmas

  

    @torch.no_grad()
    def _heun_method(
        self,
        x: Tensor,
        sigma: Tensor,
        sigma_next: Tensor,
        gamma: Tensor
    ) -> Tensor:
        epsilon = (self.s_noise**2) * torch.randn_like(x)
        sigma_hat = sigma * (1 + gamma)
        x_hat = x + (sigma_hat ** 2 - sigma ** 2)**0.5 * epsilon
        d = (x_hat - self.forward(x_hat, sigma_hat.unsqueeze(-1))[0]) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if sigma_next.values != 0:
            # Create a sigma_next tensor of the same shape as x

            model_out_next = self.forward(x_next, sigma_next.unsqueeze(-1))[0]
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * 0.5 * (d + d_prime)
        
        return x_next
    
