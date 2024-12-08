import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import math
from layers.tools.activations import get_activation_fn
import time
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
        self.beta = nn.Parameter(torch.zeros(1, dtype=torch.float32).normal_(mean=0,std=0.1))
        
    def forward(self, x):
        '''
        
            Shao, J., Hu, K., Wang, C., Xue, X., & Raj, B. (2020). 
            Is normalization indispensable for training deep neural network? 
            Advances in Neural Information Processing Systems, 33, 13434-13444. (link, pdf)
            
            
            -> normalization-free method.
        '''
        return ((1 - self.beta**2)**0.5) * x + self.beta * self.layers(x)

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
        self.latent_dim = latent_dim
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
            get_activation_fn("snake", in_chn=latent_dim),
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
        self.latent_dim = latent_dim
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


 

    def encode(self, audio :torch.Tensor, return_info=False, **kwargs):
        info = {}
        mean, std = audio.mean(dim = -1, keepdim = True), audio.std(dim = -1, keepdim = True)
        audio = (audio - mean) / std
        while (audio.dim() != 3):
            audio = audio.unsqueeze(1) if audio.dim() < 3 else audio.squeeze(1)
        latents = self.encoder(audio)

        latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
        info.update(bottleneck_info)
    
        if return_info:
            return latents, info
        return latents

    def decode(self, latents, **kwargs):
        
       
        latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents, **kwargs)
       
        while (decoded.dim() != 2):
            decoded = decoded.squeeze(1) if decoded.dim() > 2 else decoded.unsqueeze(1)
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


class AutoEncoderWrapper(nn.Module):
    def __init__(
        self,
        autoencoder = AudioAutoencoder(),
        autoencoder_state_path = None,
        ):
        super().__init__()
        self.ae = autoencoder  
        if (autoencoder_state_path is not None):
            if torch.cuda.is_available():
                print("Loaded Pretrained autoencoder. (CUDA)")
                self.ae.load_state_dict(torch.load(autoencoder_state_path))
            else:
                print("Loaded Pretrained autoencoder. (CPU)")
                self.ae.load_state_dict(torch.load(autoencoder_state_path, map_location='cpu'))
        else:
            print("Pretrained autoencoder not found. It is recommended to use pretrained autoencoder.")

        self.freeze_encoder()
        self.freeze_decoder()
        
    @torch.no_grad()
    def get_latents_shape(self, example):
        t_a = time.time()
        bs, chn, seq = self.encode_audio(example).shape
        print(f"Took {time.time() - t_a} seconds to handle {example.shape} tensor")
        return bs, chn, seq     # for 1, 16384, we have 1, 64, 4
    
    def encode(self, x):
        return self.ae.encode(x)
    
    def decode(self, x):
        return self.ae.decode(x)
    
    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        print(audio.shape)
        audio = audio.unsqueeze(1)
        if not chunked:
            # default behavior. Encode the entire audio in parallel
            return self.encode(audio)
        else:
            # CHUNKED ENCODING
            # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
            samples_per_latent = self.ae.downsampling_ratio
            total_size = audio.shape[2] # in samples
            batch_size = audio.shape[0]
            chunk_size *= samples_per_latent # converting metric in latents to samples
            overlap *= samples_per_latent # converting metric in latents to samples
            hop_size = chunk_size - overlap
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = audio[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = audio[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # Note: y_size might be a different value from the latent length used in diffusion training
            # because we can encode audio of varying lengths
            # However, the audio should've been padded to a multiple of samples_per_latent by now.
            y_size = total_size // samples_per_latent
            # Create an empty latent, we will populate it with chunks as we encode them
            y_final = torch.zeros((batch_size,self.ae.encoder.latent_dim,y_size)).to(audio.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # encode the chunk
                y_chunk = self.encode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size // samples_per_latent
                    t_end = t_start + chunk_size // samples_per_latent
                #  remove the edges of the overlaps
                ol = overlap//samples_per_latent//2
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final
    
    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Decode the entire latent in parallel
            return self.decode(latents)
        else:
            # chunked decoding
            hop_size = chunk_size - overlap
            total_size = latents.shape[2]
            batch_size = latents.shape[0]
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = latents[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = latents[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # samples_per_latent is just the downsampling ratio
            samples_per_latent = self.ae.downsampling_ratio
            # Create an empty waveform, we will populate it with chunks as decode them
            y_size = total_size * samples_per_latent
            y_final = torch.zeros((batch_size,1,y_size)).to(latents.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # decode the chunk
                y_chunk = self.decode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size * samples_per_latent
                    t_end = t_start + chunk_size * samples_per_latent
                #  remove the edges of the overlaps
                ol = (overlap//2) * samples_per_latent
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final



    def freeze_encoder(self):
        self.ae.encoder.requires_grad_(False)
        
    def freeze_decoder(self):
        self.ae.decoder.requires_grad_(False)
        
    def unfreeze_encoder(self):
        self.ae.encoder.requires_grad_(True)
        
    def unfreeze_decoder(self):
        self.ae.decoder.requires_grad_(True)
