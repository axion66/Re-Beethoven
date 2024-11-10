import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

# Karras et al. https://arxiv.org/pdf/2206.00364 implementation



class Denoiser(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        sigma_data: float=0.5,  # data distribution standard deviation
        sigma_min=0.002,
        sigma_max=1, # paper suggests 80, but I will go with 3
        rho: float = 7.0, # for image, set it 7
        s_churn: float = 40.0, # controls stochasticity(SDE)  0 for deterministic(ODE)
        s_tmin: float = 0.05, # I need to find with grid search, but who wants to do that..
        s_tmax: float = 1e+8, # Figure 15 (yellow line)
        s_noise: float = 1.003, # to inflate std for newly added noise.
        device: torch.device = torch.device("cuda:0")
    ):
        super().__init__()
        self.device = device
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_noise = lambda num_samples: (torch.randn((num_samples,1),device=device) * 1.15 - 1.2).exp()
        #torch.normal(-1.2,1.2,size=(num_samples,),device=device).exp()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max # Too high.
        self.rho = rho
        self.rho_inverse = 1.0 / rho
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_churn = s_churn
        self.s_noise = s_noise

    

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


    def forward(self, x: Tensor,sigmas=None) -> Tensor:
        # std transformation & RevIN
        # x: batch, audio_length
        b, device = x.shape[0], x.device

        #std 
        x_mean,x_std = x.mean(dim=-1,keepdim=True),x.std(dim=-1,keepdim=True)
        x = (x - x_mean) * self.sigma_data / x_std

        
        # noise
        if sigmas is None:
            sigmas = self.sigma_noise(num_samples=b)
        x_noised = x + (torch.randn_like(x) * sigmas) # randn_like * sigmas == noise

        c_skip, c_out, c_in, c_noise = [self.append_dims(x, x.ndim) for x in self.get_scalings(sigmas)]
        x_denoised = self.model(c_in * x_noised, c_noise) * c_out + x * c_skip
        
        #std
        x_denoised = x_denoised * x_std / self.sigma_data + x_mean
        return x_denoised, sigmas

    def loss_fn(self,x:Tensor):
 
        b, device = x.shape[0], x.device

        # std transform
        x_mean,x_std = x.mean(dim=-1,keepdim=True),x.std(dim=-1,keepdim=True)
        x = (x - x_mean) * self.sigma_data / x_std
            
        # noise
        #mask = torch.rand_like(x) < 0.95  #70 will be noise, while other 30 will be unnoised.
        sigmas = self.sigma_noise(num_samples=b)
        noise = torch.randn_like(x) * sigmas
        x_noised = x + (torch.randn_like(x) * sigmas) # randn_like * sigmas == noise


        c_skip, c_out, c_in, c_noise = [self.append_dims(x, x.ndim) for x in self.get_scalings(sigmas)]
        x_denoised = self.model(c_in * x_noised, c_noise)#   * c_out + x * c_skip -> replaced by changing original x.
        x = (x - c_skip * x_noised) / c_out # instead of transforming the x_denoised, we transform the original x.

        loss = self._weighting_snr(sigmas) * ((x_denoised - x)**2)
        #print(f"weight: {self._weighting_snr(sigmas)} \n loss: {loss}, \n w/o weight: {loss / self._weighting_snr(sigmas)}")
        return loss.reshape(-1).mean()


    def _weighting_snr(self, sigmas):
        #return (sigmas ** 2 + self.sigma_data ** 2) / (sigmas * self.sigma_data) ** 2 -> crazy-level magnitude.(0.5 to 1000)
        return self.sigma_data ** 2 / (sigmas ** 2 + self.sigma_data ** 2) #(snr)
        #return 1

    # sampling part
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int,
    ) -> Tensor:
        

        sigmas = self._schudule_sigmas(num_steps).unsqueeze(-1) # t = {batch,1}

        x = sigmas[0] ** 2 * torch.randn((num_samples,self.model.sequence_length),device=self.device) 
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
            self.sigma_max ** self.rho_inverse
            + (steps / (num_steps - 1)) * (self.sigma_min ** self.rho_inverse - self.sigma_max ** self.rho_inverse)
        ) ** self.rho
        # sigmas maximum -> minimum, as sampling method goes backward(T to 0)
        # Although original paper suggested maximum=80, We should go with maximum=0.8~3, as that's expected noise range used in training step is around there. (Also to reduce cost)
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
        d = (x_hat - self.forward(x_hat, sigma_hat)[0]) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if sigma_next.values != 0:
            # Create a sigma_next tensor of the same shape as x

            model_out_next = self.forward(x_next, sigma_next)[0]
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * 0.5 * (d + d_prime)
        
        return x_next
    
