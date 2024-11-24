
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