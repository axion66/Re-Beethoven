import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from layers.preprocess import load_mp3_files, create_overlapping_chunks_tensor
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import soundfile as sf  
from tqdm import tqdm
import wandb
import argparse
import pytorch_warmup as warmup
from layers.autoencoder.vae import AudioAutoencoder
from layers.tools.losses import *
class Trainer:
    def __init__(self, cfg_path):
        with open(cfg_path) as stream:
            self.cfg = yaml.safe_load(stream)
            self.FFT_CFG = self.cfg['fft']
            self.MODEL_CFG = self.cfg['model']
            self.FILE_CFG = self.cfg['file']

        '''MODEL'''
        self.net = AudioAutoencoder(
            sample_rate=self.FFT_CFG['sr'],
            downsampling_ratio=self.FFT_CFG['downsampling_ratio']
        ).to(self.MODEL_CFG['device'])
        
        self.loss = LossModule(name="AutoEncoder")
        
        # This one was the problemtic one. I need to train this crazy-ass Discriminator, not use it as a loss!!!!
        #self.loss.append("discriminator", weight_loss=1.0, module=)
        
        self.loss.append("mrstft", weight_loss=3.0, module=MultiResolutionSTFTLoss(
            sample_rate=self.FFT_CFG['sr']).to(self.MODEL_CFG['device']))
        #self.loss.append("L1", weight_loss=0.1,module=L1Loss().to(self.MODEL_CFG['device']))
        
        
        '''Loader'''
        self.train_loader, self.test_loader = self.getLoader()
        
        
        warmup_period = self.MODEL_CFG['warmup_period']
        num_steps = len(self.train_loader) * self.MODEL_CFG['epoch'] - warmup_period

        self.discriminator = EncodecDiscriminator().to(self.MODEL_CFG['device'])
        self.optim_dis = Adam(self.discriminator.parameters(), lr=self.MODEL_CFG['lr'])
        self.lr_schedule_dis = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_dis, T_max=num_steps)
        self.warmup_schedule_dis = warmup.ExponentialWarmup(self.optim_dis, warmup_period)

        # In __init__
        self.optim_gen = Adam(self.net.parameters(), lr=self.MODEL_CFG['lr'])
        self.lr_schedule_gen = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_gen, T_max=num_steps)
        self.warmup_schedule_gen = warmup.ExponentialWarmup(self.optim_gen, warmup_period)
        
        
        '''LOG'''
        self.train_losses = []
        self.eval_losses = []
        self.best_eval_loss = float('inf')

        
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
        for name, param in self.net.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}") 
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)  
        print(f"Total trainable parameters: {total_params:,}")
        
        EPOCH = self.MODEL_CFG['epoch']
        LOG_DIR = self.FILE_CFG['log_dir']
        os.makedirs(LOG_DIR, exist_ok=True)


        for epoch in range(self.MODEL_CFG['epoch']):
            
            EPOCH_LOSS = []

            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCH}")):
                
                
                x = x[0].to(self.MODEL_CFG['device']).squeeze(1)
                latents = self.net.encode(x)
                decoded = self.net.decode(latents)

                
                
                if (self.loss.discriminator_step % 2 == 0):
                    '''
                        note that discriminator itself is a trainable model.
                            The generator(adv_loss, feature_matching_loss) is connected w/ the diffusion model,
                            while the dis_loss is also connected w/ the diffusion model indirectly.
                    '''
                    loss, _, _ = self.discriminator.loss(decoded.unsqueeze(1),x.unsqueeze(1))
                    self.optim_dis.zero_grad()
                    loss.backward()
                    self.optim_dis.step()
                    with self.warmup_schedule_dis.dampening():
                        if self.warmup_schedule_dis.last_step + 1 >= self.MODEL_CFG['warmup_period']:
                            self.lr_schedule_dis.step()
                    info = {
                        "discriminator/dis_loss" : loss
                    }
                
                else:
                    loss, info = self.loss.loss_fn(decoded, x)
                    _, adv_loss, feature_matching_loss = self.discriminator.loss(decoded.unsqueeze(1), x.unsqueeze(1))
                    loss = loss + adv_loss + feature_matching_loss
                    info.update(
                        {
                            'discriminator/adv_loss':adv_loss,
                            "discriminator/feature_matching_loss": feature_matching_loss
                        }
                    )
                    
                    self.optim_gen.zero_grad()
                    loss.backward()
                    self.optim_gen.step()
                    with self.warmup_schedule_gen.dampening():
                        if self.warmup_schedule_gen.last_step + 1 >= self.MODEL_CFG['warmup_period']:
                            self.lr_schedule_gen.step()
                            
                self.loss.discriminator_step += 1
                        
                # Timestamp Loss, LR
                EPOCH_LOSS.append(loss.item())
                wandb.log({"timestamp_loss": loss, 
                           "lr": self.optim_gen.param_groups[0]['lr'],
                           })
                wandb.log(info)

            # Epoch Loss
            avg_train_loss = sum(EPOCH_LOSS) / len(EPOCH_LOSS)
            self.train_losses.append(avg_train_loss)
            wandb.log({"train_loss": avg_train_loss})
            print(f"Epoch {epoch+1}/{EPOCH}, Average Train Loss: {avg_train_loss:.4f}")

            # Evaluation & Sampling
            if (epoch + 1) % self.MODEL_CFG['evaluation_cycle'] == 0:
                self.net.eval()
                EVAL_LOSS = []
                with torch.no_grad():
                    STORE_SAMPLE_DIR = os.path.join(LOG_DIR, f"samples_epoch_{epoch+1}")
                    for idx,x in enumerate(self.test_loader):
                        x = x[0].to(self.MODEL_CFG['device']) # list to TEnsor
                        latents = self.net.encode(x)
                        decoded = self.net.decode(latents)
                        
                        loss,_ = self.loss.loss_fn(decoded,x)
                        
                        
                        self.generate_samples(decoded,output_dir=STORE_SAMPLE_DIR)
                        EVAL_LOSS.append(loss.item())
            
                avg_eval_loss = sum(EVAL_LOSS) / len(EVAL_LOSS)
                self.eval_losses.append(avg_eval_loss)
                wandb.log({"eval_loss": avg_eval_loss})
                print(f"Evaluation Loss: {avg_eval_loss:.4f}")
                
                
                if avg_eval_loss < self.best_eval_loss:
                    self.best_eval_loss = avg_eval_loss
                    torch.save(self.net.state_dict(), os.path.join(LOG_DIR, 'best_model.pth'))
                '''
                STORE_SAMPLE_DIR = os.path.join(LOG_DIR, f"samples_epoch_{epoch+1}")
                for idx in range(self.MODEL_CFG['num_samples']):
                    self.generate_samples(
                        num_samples=1,#self.MODEL_CFG['num_samples'],
                        num_steps=self.MODEL_CFG['sampling_steps'][idx],
                        output_dir=STORE_SAMPLE_DIR,
                        idx=idx
                    )

                for i in range(self.MODEL_CFG['num_samples']):
                    sample_path = os.path.join(STORE_SAMPLE_DIR, f"Sample_{i}.wav")
                    wandb.log({f"Sample {i}": wandb.Audio(sample_path, sample_rate=self.FFT_CFG['sr'])})
                '''
                self.net.train()

            



    def getLoader(self):
        tensors = load_mp3_files(base_folder=self.FILE_CFG['audio_folder'], config=self.FFT_CFG)
        tensors = torch.cat(tensors, dim=-1)

        x = create_overlapping_chunks_tensor(sequence=tensors, config=self.FFT_CFG)
        x = x[torch.randperm(x.size(0))]

        train = TensorDataset(x[:-self.FFT_CFG['num_evaluation'], :])
        test = TensorDataset(x[-self.FFT_CFG['num_evaluation']:, :])
        trainLoader = DataLoader(train, batch_size=self.MODEL_CFG['batch_size'], num_workers=self.MODEL_CFG['num_workers'], shuffle=True)
        testLoader = DataLoader(test, batch_size=self.MODEL_CFG['batch_size'], num_workers=self.MODEL_CFG['num_workers'], shuffle=False)


        return trainLoader, testLoader


    def generate_samples(
            self,
            x,
            output_dir: str,
        ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        for i in range(x.size(0)):
            sample = x[i].cpu().numpy().flatten()
            filename = os.path.join(output_dir, f"Sample_{i}.wav")
            sf.write(filename, sample, self.FFT_CFG['sr'])
            wandb.log({f"Sample {i}": wandb.Audio(sample, sample_rate=self.FFT_CFG['sr'])})

     

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    trainer = Trainer(args.cfg_path)
    trainer.train()