import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from torch.autograd import Variable

# Karras et al. https://arxiv.org/pdf/2206.00364 implementation



class GANWrapper(nn.Module):

    def __init__(
        self,
        config,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device = torch.device("cuda:0")
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.advLoss = torch.nn.BCELoss()




    def loss_fn(
            self,
            x:Tensor,
            optimizer_generator,
            optimizer_discriminator):
        

        x_mean,x_std = x.mean(dim=-1,keepdim=True),x.std(dim=-1,keepdim=True)
        x = (x - x_mean) / x_std

        optimizer_generator.zero_grad()
        real_output = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False) 
        fake_output = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)
        x_generated = self.generator(torch.randn_like(x,device=self.device),torch.zeros((x.size(0),1),device=self.device))
        generator_loss = self.advLoss(self.discriminator(x_generated,torch.zeros((x.size(0),1),device=self.device)), real_output)
        generator_loss.backward()
        optimizer_generator.step()   


        optimizer_discriminator.zero_grad()
        discriminator_loss_real = self.advLoss(self.discriminator(x,torch.zeros((x.size(0),1),device=self.device)), real_output)
        discriminator_loss_fake = self.advLoss(self.discriminator(x_generated.detach(),torch.zeros((x.size(0),1),device=self.device)), fake_output)
        discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
        discriminator_loss.backward()
        optimizer_discriminator.step()

        return generator_loss.item(), discriminator_loss.item(), discriminator_loss_real.item(),discriminator_loss_fake.item()


    # sampling part
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
    ) -> Tensor:
        
        x = torch.randn((num_samples,self.config['seq_len']),device=self.device)

        return self.generator(x)    
    
 
