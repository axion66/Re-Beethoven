import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attn import DiffMHAFlash,TransformerBlock,TimestepBlockA
import math
from layers.tools.utils import Linear,exists
from abc import abstractmethod




class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
 
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,

    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm


        self.in_layers = nn.Sequential(
            nn.GroupNorm(32,channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32,self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1) # if out_chn is different, align input_chn with out_chn.

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            h = self.in_layers[:-1](x)
            h = self.h_upd(h)
            h = self.in_layers[-1](h)

            x = self.x_upd(x)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): # expand dim
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, chunks=2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h
    

  



class TimestepEmbedSequential(nn.Sequential, TimestepBlock,TimestepBlockA):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock) or isinstance(layer,TimestepBlockA):
                x = layer(x, emb)
            else:
                x = layer(x)
              
        return x

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class transpose_norm_fn(TimestepBlock):
    """
    A Timestep class that wraps a normalization function with transpose operations.
    """

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x, emb):
        x = x.transpose(1,-1)
        x = self.norm(x, emb)
        return x.transpose(1,-1)
    

# UNET MODEL


class net(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.



            batch_size = 1
            seq_len = 256
            channels = 512

            # Create model
            model = net(
                in_channels=512,               # e.g., for RGB images
                model_channels=64,           # Base number of channels
                out_channels=512,              # Typically same as input for autoencoders
                num_res_blocks=2,            # Number of ResBlocks per level
                attention_resolutions=[4, 8],# Apply attention at 1/4 and 1/8 resolutions
                dropout=0.1,                 # Dropout rate
                channel_mult=(1, 2, 4, 8),   # Channel multiplier for each level
                conv_resample=True,          # Use convolutional down/upsampling
                dims=1,                      # 2D data (e.g., images)
                use_checkpoint=False,        # Gradient checkpointing to save memory
                use_fp16=False,              # Use float16 for memory efficiency
                num_heads=4,                 # Attention heads for TransformerBlock
                num_head_channels=32,        # Channels per head
                use_scale_shift_norm=True,   # Use scale-shift normalization
                resblock_updown=True         # Use ResBlock for up/downsampling
            )

            # Create input tensor [B, T, C]
            x = torch.randn(batch_size, seq_len, channels)
            # Create timesteps
            timesteps = torch.randint(0, 1000, (batch_size,1))
            # Forward pass
            # Need to transpose input to [B, C, T] for convolutions
            x = x.transpose(1, 2)  # [B, C, T]
            output = model(x, timesteps)

    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_fp16=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        gan_mode=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            FourierFeatures(in_features=1,out_features=model_channels),
            Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        transpose_norm_fn(TransformerBlock(
                            ch,
                            depth=level+1,
                            num_heads=num_heads,
                        ))
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            transpose_norm_fn(TransformerBlock(
                ch,
                depth=1,
                num_heads=num_heads,
            )),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        transpose_norm_fn(TransformerBlock(
                            ch,
                            depth=level+1,
                            num_heads=num_heads_upsample,
                        )),
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(8,ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        self.gan_mode = gan_mode
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(self.convert_module_to_f16)
        self.middle_block.apply(self.convert_module_to_f16)
        self.output_blocks.apply(self.convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(self.convert_module_to_f32)
        self.middle_block.apply(self.convert_module_to_f32)
        self.output_blocks.apply(self.convert_module_to_f32)

    def convert_module_to_f16(self,l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()


    def convert_module_to_f32(self,l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()



    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        b,l = x.shape
        assert l % self.in_channels == 0
        x = x.reshape(b, self.in_channels, l//self.in_channels)
        hs = []
        emb = self.time_embed(timesteps)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        i=0
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            i+=1
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        if self.gan_mode:
            return self.out(h).reshape(b,-1).mean(-1)
        return self.out(h).reshape(b,-1)










"""
    OLD CNN Components:

    class ResBlock(nn.Module):
    def __init__(
        self, 
        channels: int,
        kernel: int = 3,
        norm_groups: int = 4,
        dilation: int = 1, # 1, 2, 4, 8 ...
        norm_fn=None,
        activation_fn=None,
        dropout: float = 0.
    ):
        super().__init__()
        assert kernel % 2 == 1, "Kernel size should be odd to maintain shape."
        padding = dilation * (kernel - 1) // 2

        # 1st Conv
        self.norm1 = transpose_norm_fn(norm_fn(channels)) if exists(norm_fn) else nn.GroupNorm(norm_groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel, 1, padding, dilation=dilation)
        self.activation1 = activation_fn if exists(activation_fn) else nn.SiLU()
        self.dropout1 = nn.Dropout1d(dropout)


        # 2nd Conv
        self.norm2 = transpose_norm_fn(norm_fn(channels)) if exists(norm_fn) else nn.GroupNorm(norm_groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, 1, padding, dilation=dilation)
        self.activation2 = activation_fn if exists(activation_fn) else nn.SiLU()
        self.dropout2 = nn.Dropout1d(dropout)

    
    def forward(self, x):
        h = self.dropout1(self.conv1(self.activation1(self.norm1(x))))
        h = self.dropout2(self.conv2(self.activation2(self.norm2(h))))
        return x + h # require norm after ResNet-stack is passed.


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_fn=None,
        activation_fn=None,
        p=0.2,
        mode:str="encode_or_decode"
    ):
        super().__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv1d(in_channels=in_channels, 
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True) 
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose1d(in_channels=in_channels, 
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True) 
        else:
            raise Exception(f"Conv Mode not defined: {mode}")
        self.ln1 = transpose_norm_fn(norm_fn(out_channels)) if exists(norm_fn) else nn.GroupNorm(4,out_channels)
        self.activation = transpose_norm_fn(activation_fn) if exists(activation_fn) else nn.SiLU()
        self.conv2 = ResBlock(channels=out_channels,
                              kernel=3,
                              activation_fn=activation_fn,
                              norm_fn=norm_fn,
                              dropout=p)
   
        self.ln2 = transpose_norm_fn(norm_fn(out_channels)) if exists(norm_fn) else nn.GroupNorm(4,out_channels)
    def forward(self, x):
        # Input shape: [batch, length, channels]
        x = x.transpose(-1, -2)  # Change to [batch, channels, length]
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.ln2(x)
        return x.transpose(-1, -2) 

class Encoder(nn.Module):
    
    # turn freq_bins into latent vectors.
    
    def __init__(
        self,
        channels:list = [1,2,3,4],
        activation_fn=None,
        norm_fn=None,
        p=0.2,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [Block(mode='encode',in_channels=channels[i], out_channels=channels[i + 1],activation_fn=activation_fn,norm_fn=norm_fn, p=p) for i in range(len(channels) - 1)]
        )

        
    def forward(self,x):
        
        # Input: Batch, length, channels
        
        for block in self.blocks:
            x = block(x)

        return x
    

class Decoder(nn.Module):
    
    # Turn latent vectors back into freq_bins.
    
    def __init__(
        self,
        channels: list = [4, 3, 2, 1],
        activation_fn=None,
        norm_fn=None,
        p=0.2,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [Block(mode='decode',in_channels=channels[i], out_channels=channels[i + 1], 
                           activation_fn=activation_fn, norm_fn=norm_fn, p=p) for i in range(len(channels) - 1)]
        )

    def forward(self, x):
        
        # Input: Batch, length, channels
        
        for block in self.blocks:
            x = block(x)

        return x
"""