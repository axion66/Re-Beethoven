import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from layers.tools.audios import RevSTFT
from layers.tools.utils import *
from layers.attn import TransformerBlock
from layers.cnn import Encoder,Decoder,ResBlock
from layers.tools.activations import get_activation_fn
from layers.tools.norms import get_norm_fn
#from nnAudio.features import STFT,iSTFT

class FourierFeatures(nn.Module):
    # from NCSN++.
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std) 

    def forward(self, x):
        f = 2 * 3.141592653589793 * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)




class VQTokenizer(nn.Module):
    def __init__(self,config):
        self.device = torch.device(config['device'])
        self.config_path = "WavTokenizer/configs/medium_matadata.yml"
        self.model_path = "../pretrained_models/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
        #audio_outpath = "xxx"

        self.wavtokenizer = WavTokenizer.from_pretrained0802(self.config_path, self.model_path).to(self.device)
        self.freeze_model(self.wavtokenizer)

        self.bandwidth_id = torch.tensor([0])

    def freeze_model(self,model):
        for param in model.parameters():
            param.requires_grad = False


    def encode(self,wav):
        """
            wav:Tensor should have sr == 24000
        """
        #wav, sr = torchaudio.load("../../../dataset/no8/0/audio0.mp3")
        #wav = convert_audio(wav, sr, 24000, 1) 
        
        wav=wav.to(self.device)
        features,discrete_code= self.wavtokenizer.encode_infer(wav, bandwidth_id=self.bandwidth_id)
        
        return features,discrete_code
    def decode(self,features):
        audio_out = self.wavtokenizer.decode(features, bandwidth_id=self.bandwidth_id) 
        #torchaudio.save(audio_outpath, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
        return audio_out
    
class net(nn.Module):
   
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.sequence_length = config['seq_len']                                                # Raw sequence length
        self.seq_len,self.embed_dim = 1000,16 # for 240,000 length(10sec) audio
        self.num_blocks = config['num_blocks']                                                  # Number of Transformer blocks
        activation_fn = get_activation_fn(config['activation_fn'],in_chn=self.embed_dim)
        norm_fn = get_norm_fn(config['norm_fn'])
        p = config['dropout']
          
        
        # Mapping Net
        self.time_emb = FourierFeatures(1, 512//8//2)
        self.map_layers = nn.Sequential(
            Linear(512//8//2, 512//8//2), # head_dim
            activation_fn,
            Linear(512//8//2,512//8//2),
            activation_fn,
            Linear(512//8//2,512//8//2)
        )

        self.encoder = Encoder(channels=[self.embed_dim,512,512],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.decoder = Decoder(channels=[512,512,self.embed_dim],activation_fn=activation_fn,norm_fn=norm_fn,p=p)
        self.transformer = nn.ModuleList(
            [TransformerBlock(embed_dim=512, depth=i + 1, num_heads=8,activation_fn=activation_fn,norm_fn=norm_fn) for i in range(self.num_blocks)]
        )
    
        self.last = ResBlock(channels=self.embed_dim,
                              kernel=3,
                              activation_fn=activation_fn,
                              norm_fn=norm_fn,
                              dropout=0)
        


    def forward(self,x,sigmas): 
        '''
            x: [batch,seq],
            sigmas: [batch]
        '''
        # Condition Mapping (Timestamp)
        sigmas = self.time_emb(sigmas.unsqueeze(-1))
        sigmas = self.map_layers(sigmas)
        # Condition Mapping
        
        x = x.reshape(x.size(0), self.seq_len, self.embed_dim)        

        x = self.encoder(x)
        
        for trans in self.transformer:
            x = trans(x,sigmas)

        x = self.decoder(x)
  
        x = x.transpose(-1,-2)
        x = self.last(x)
        x = x.transpose(-1,-2)


        return x.reshape(x.size(0),-1)
    



