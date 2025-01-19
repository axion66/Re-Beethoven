import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW,Adam
from layers.tools.preprocess import load_files, create_overlapping_chunks_tensor
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import soundfile as sf  
from tqdm import tqdm
import wandb
import argparse
import pytorch_warmup as warmup
from layers.autoencoder.vae import AudioAutoencoder,AEBottleneck
from layers.tools.losses import *


def print_model_size(name, model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()}") 
            
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"{name} | Trainable[requires_grad] parameters: {total_params}")
    
    
class Trainer:
    def __init__(self, cfg_path):
        
        self.best_eval_loss = 777_777_777
        self.configure_config(cfg_path)
        self.configure_model()
        self.configure_loss()
        self.configure_loader()
        self.configure_optimizer()
        self.configure_wandb()
        
    
    def configure_config(self, cfg_path):
        with open(cfg_path) as stream:
            self.cfg = yaml.safe_load(stream)
            self.FFT_CFG = self.cfg['fft']
            self.MODEL_CFG = self.cfg['model']
            self.FILE_CFG = self.cfg['file']
            
    def configure_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = AudioAutoencoder(
            sample_rate = self.FFT_CFG['sr'],
            downsampling_ratio = self.FFT_CFG['downsampling_ratio'],
            bottleneck = AEBottleneck()
        ).to(self.device)
        
        if (self.MODEL_CFG['pretrained_autoencoder'] != False):
            print(f"Loaded pretrained model at {self.MODEL_CFG['pretrained_autoencoder']} | File exists: {os.path.exists(self.MODEL_CFG['pretrained_autoencoder'])}")
            self.net.load_state_dict(torch.load(self.MODEL_CFG['pretrained_autoencoder']))
            
        if self.MODEL_CFG['freeze_encoder']:
            print("Encoder freezed.")
            for layer in self.net.encoder.parameters():
                layer.requires_grad = False
        
        if self.MODEL_CFG['freeze_decoder']:
            print("Decoder freezed.")
            for layer in self.net.decoder.parameters():
                layer.requires_grad = False
        print_model_size("Autoencoder", self.net)
        
    def configure_loss(self):
        '''
            Configure Only generator Loss. Discriminator loss is configured at (configure_optimizer).
        '''
        self.loss = LossModule(name="AutoEncoder")
        self.loss.append("mrstft", weight_loss=0.25, module=MultiResolutionSTFTLoss(
            sample_rate=self.FFT_CFG['sr']).to(self.device))
        #self.loss.append("L1", weight_loss=0.1,module=L1Loss().to(self.MODEL_CFG['device']))        

    
    def configure_loader(self):
        tensors = load_files(base_folder=self.FILE_CFG['audio_folder'], config=self.FFT_CFG)
        tensors = torch.cat(tensors, dim=-1)

        x = create_overlapping_chunks_tensor(sequence=tensors, config=self.FFT_CFG)
        x = x[torch.randperm(x.size(0))]

        train = TensorDataset(x[:-self.FFT_CFG['num_evaluation'], :])
        test = TensorDataset(x[-self.FFT_CFG['num_evaluation']:, :])
        self.train_loader = DataLoader(train, batch_size=self.MODEL_CFG['batch_size'], num_workers=self.MODEL_CFG['num_workers'], shuffle=True)
        self.test_loader = DataLoader(test, batch_size=self.MODEL_CFG['batch_size'], num_workers=self.MODEL_CFG['num_workers'], shuffle=False)


    def configure_optimizer(self):
        warmup_period = self.MODEL_CFG['warmup_period']
        #self.discriminator = EncodecDiscriminator().to(self.device)
        #self.optim_dis = AdamW(self.discriminator.parameters(), lr=self.MODEL_CFG['lr'] * 2, betas=(0.8,0.99))
        #self.lr_schedule_dis = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_dis, T_max=num_steps)
        #self.warmup_schedule_dis = warmup.ExponentialWarmup(self.optim_dis, warmup_period)
        
        self.optim_gen = AdamW(self.net.parameters(), lr=self.MODEL_CFG['lr'], betas=(0.8,0.99))
        #self.lr_schedule_gen = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_gen, T_max=num_steps)
        #self.warmup_schedule_gen = warmup.ExponentialWarmup(self.optim_gen, warmup_period)
        
        
    def configure_wandb(self):
        wandb_store = os.path.join(self.FILE_CFG['log_dir'],"wandb_store")
        os.makedirs(wandb_store,exist_ok=True)
       
        wandb.init(
            project=f"autoencoder_{self.FILE_CFG['project_name']}",
            name = f"{self.FILE_CFG['run_name']}_{datetime.now().strftime('%m%d_%H-%M')}",
            dir=wandb_store,
            config={
                "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected.",
                **self.cfg
            },
            
        )
    
    
    def train(self):
        # No Discriminator using-mode.
        
        EPOCH = self.MODEL_CFG['epoch']
        LOG_DIR = self.FILE_CFG['log_dir']
        os.makedirs(LOG_DIR, exist_ok=True)

        step = 0
        for epoch in range(self.MODEL_CFG['epoch']):
            
            epochloss = []

            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCH}")):
                info = {}
                x = x[0].to(self.device).unsqueeze(1)  # batch, 1, seq
                latents = self.net.encode(x)
                decoded = self.net.decode(latents).unsqueeze(1) # batch, 1, seq

                
                #dis_loss, adv_loss, feature_matching_loss = self.discriminator.loss(x, decoded)
                '''
                if (self.loss.discriminator_step % 2 == 0):
                    if (dis_loss < 1e+5):
                        self.optim_dis.zero_grad()
                        dis_loss.backward()
                        self.optim_dis.step()
                        
                        info.update({"discriminator/dis_loss" : dis_loss})
                else:'''
                
                loss, info_loss_fn = self.loss.loss_fn(x, decoded)
                gen_loss = loss# + adv_loss + feature_matching_loss

                if gen_loss < 1e+5:
                    self.optim_gen.zero_grad()
                    gen_loss.backward()
                    self.optim_gen.step()
                    
                    info.update(
                        {
                            **info_loss_fn,
                            "generator_loss": gen_loss,
                            #"adv_loss":adv_loss,
                            #"feature_matching_loss": feature_matching_loss
                        }
                    )
                    epochloss.append(gen_loss.item()) 
                          
                self.loss.discriminator_step += 1
                       
                info.update({
                    "lr/lr_gen": self.optim_gen.param_groups[0]['lr'],
                    }
                ) 
               
                wandb.log(info)
                
                step += 1
                if step % self.MODEL_CFG['evaluation_cycle'] == 0:
                    self.net.eval()
                    with torch.no_grad():
                        EVAL_LOSS = []
                        FAKE_DIR = os.path.join(LOG_DIR, f"generated_epoch_{epoch+1}")
                        ORIG_DIR = os.path.join(LOG_DIR, f"original_epoch_{epoch+1}")
                        os.makedirs(FAKE_DIR, exist_ok=True)
                        os.makedirs(ORIG_DIR, exist_ok=True)
                        for idx, x in enumerate(self.test_loader):
                            x = x[0].to(self.device)
                            latents = self.net.encode(x)
                            decoded = self.net.decode(latents)
                            loss,_ = self.loss.loss_fn(x.unsqueeze(1), decoded.unsqueeze(1))
                            
                            self.generate_samples(decoded,output_dir=FAKE_DIR, name="fake")
                            self.generate_samples(x,output_dir=ORIG_DIR, name="real")
                            
                            EVAL_LOSS.append(loss.detach().cpu().item())
    
                    avg_eval_loss = sum(EVAL_LOSS) / len(EVAL_LOSS)
                    wandb.log({"eval_loss": avg_eval_loss})
                    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
                    
                    if avg_eval_loss < self.best_eval_loss:
                        self.best_eval_loss = avg_eval_loss
                        torch.save(self.net.state_dict(), os.path.join(LOG_DIR, 'best_model.pth'))
                        print(f"Best model saved at {LOG_DIR}/best_model.pth")
                    self.net.train()
                    '''
                    for name, module in self.net.named_modules():
                            if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, "grad"):
                                wandb.log({f"gradients/{name}.weight": wandb.Histogram(module.weight.grad.cpu().numpy())})
                            if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, "grad"):                   
                                wandb.log({f"gradients/{name}.bias": wandb.Histogram(module.bias.grad.cpu().numpy())})
                           

                    '''
                
            #wandb.log({"Average Generator Loss": sum(epochloss) / len(epochloss)})
            print(f"Epoch {epoch+1}/{EPOCH}, Average Generator Loss: {sum(epochloss) / len(epochloss)}")


            


    


    def generate_samples(
            self,
            x,
            output_dir: str,
            name: str,
        ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for i in range(x.size(0)):
            sample = x[i].cpu().numpy().flatten()
            filename = os.path.join(output_dir, f"Sample_{i}.wav")
            sf.write(filename, sample, self.FFT_CFG['sr'])
            wandb.log({f"{name}/Sample {i}": wandb.Audio(sample, sample_rate=self.FFT_CFG['sr'])})

     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    trainer = Trainer(args.cfg_path)
    trainer.train()