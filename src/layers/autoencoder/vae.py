import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from layers.tools.activations import get_activation_fn

# from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/autoencoders.py and modified

class TemporalResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation):
        super().__init__()

        self.layers = nn.Sequential(
            get_activation_fn("snake", in_chn=out_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=(dilation * (7-1)) // 2),
            get_activation_fn("snake", in_chn=out_channels),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride):

        super().__init__()

        self.l = nn.Sequential(
            TemporalResBlock(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            TemporalResBlock(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            TemporalResBlock(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            get_activation_fn("snake",in_chn=in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.l(x)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride):
        super().__init__()
        # upsample + conv > transposedConv -> due to artifacts in transposed conv.
        self.layers = nn.Sequential(
            get_activation_fn("snake", in_chn=in_channels),
            nn.Upsample(scale_factor=stride, mode="nearest"),
            nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels, 
                    kernel_size=2*stride,
                    stride=1,
                    bias=False,
                    padding='same'),
            TemporalResBlock(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            TemporalResBlock(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            TemporalResBlock(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),
        )

    def forward(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 channels=128, 
                 latent_dim=128, 
                 c_mults = [1, 2, 4, 8, 16],    # self.depth is [1] + len(c_mults)
                 strides = [2, 4, 8, 8, 8],
        ):
        super().__init__()
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            nn.Conv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i])]

        layers += [
            get_activation_fn("snake", in_chn=c_mults[-1] * channels),
            nn.Conv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, 
                 out_channels=1, 
                 channels=128, 
                 latent_dim=128, 
                 c_mults = [1, 2, 4, 8, 16], 
                 strides = [2, 4, 8, 8, 8],):
               
        super().__init__()

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            nn.Conv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                )
            ]

        layers += [
            get_activation_fn("snake", in_chn=c_mults[0] * channels),
            nn.Conv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VAEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def vae_sample(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * std + mean
        latents = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        return latents, kl
    
    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = self.vae_sample(mean, scale)

        info["kl"] = kl
        if return_info:
            return x, info
        else:
            return x

    def decode(self,x):
        return x
class AEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        
  
    def encode(self, x, return_info=False, **kwargs):
        
        if return_info:
            return x, {}
        else:
            return x
    def decode(self,x):
        return x


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder=OobleckEncoder(),
        decoder=OobleckDecoder(),
        # chn increases 1 -> 128 -> 256 ... 2048 -> 32(VAE) -> 32(VAE) -> 2048 -> ... 1
        downsampling_ratio=2048,
        sample_rate=8000,
        io_channels=1,
        bottleneck = AEBottleneck(),
        in_channels = 1,
        out_channels = 1,
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder


 

    def encode(self, audio, return_info=False, **kwargs):

        info = {}
        latents = self.encoder(audio)

        latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
        info.update(bottleneck_info)
    
        if return_info:
            return latents, info
        return latents

    def decode(self, latents, **kwargs):
        
       
        latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents, **kwargs)
       
        return decoded
   
    

'''
class DiffusionAutoencoder(AudioAutoencoder):
    def __init__(
        self,
        diffusion: ConditionedDiffusionModel,
        diffusion_downsampling_ratio,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        self.min_length = self.downsampling_ratio * diffusion_downsampling_ratio

        if self.encoder is not None:
            # Shrink the initial encoder parameters to avoid saturated latents
            with torch.no_grad():
                for param in self.encoder.parameters():
                    param *= 0.5

    def decode(self, latents, steps=100):

        upsampled_length = latents.shape[2] * self.downsampling_ratio

        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        if self.decoder is not None:
            latents = self.decode(latents)
    
        # Upsample latents to match diffusion length
        if latents.shape[2] != upsampled_length:
            latents = F.interpolate(latents, size=upsampled_length, mode='nearest')

        noise = torch.randn(latents.shape[0], self.io_channels, upsampled_length, device=latents.device)
        decoded = sample(self.diffusion, noise, steps, 0, input_concat_cond=latents)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    decoded = self.pretransform.decode(decoded)

        return decoded
    
def create_diffAE_from_config(config: Dict[str, Any]):
    
    diffae_config = config["model"]

    if "encoder" in diffae_config:
        encoder = create_encoder_from_config(diffae_config["encoder"])
    else:
        encoder = None

    if "decoder" in diffae_config:
        decoder = create_decoder_from_config(diffae_config["decoder"])
    else:
        decoder = None

    diffusion_model_type = diffae_config["diffusion"]["type"]

    if diffusion_model_type == "DAU1d":
        diffusion = DAU1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "adp_1d":
        diffusion = UNet1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "dit":
        diffusion = DiTWrapper(**diffae_config["diffusion"]["config"])

    latent_dim = diffae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = diffae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = diffae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    bottleneck = diffae_config.get("bottleneck", None)

    pretransform = diffae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    diffusion_downsampling_ratio = None,

    if diffusion_model_type == "DAU1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["strides"])
    elif diffusion_model_type == "adp_1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["factors"])
    elif diffusion_model_type == "dit":
        diffusion_downsampling_ratio = 1

    return DiffusionAutoencoder(
        encoder=encoder,
        decoder=decoder,
        diffusion=diffusion,
        io_channels=io_channels,
        sample_rate=sample_rate,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        diffusion_downsampling_ratio=diffusion_downsampling_ratio,
        bottleneck=bottleneck,
        pretransform=pretransform
    )
'''