import os
import yaml
import torch
import torch.nn as nn
from layers.diffusion.karras import Denoiser
from torch.optim import Adam
from layers.preprocess import load_mp3_files, create_overlapping_chunks_tensor
from torch.utils.data import TensorDataset, DataLoader
import wandb
from datetime import datetime
import soundfile as sf  
from tqdm import tqdm
import argparse
import pytorch_warmup as warmup

from layers.autoencoder.vae import AudioAutoencoder
from layers.tools.losses import MultiResolutionSTFTLoss, RawL1Loss, EncodecDiscriminator

class Trainer:
    def __init__(self, cfg_path):
        with open(cfg_path) as stream:
            self.cfg = yaml.safe_load(stream)
            self.FFT_CFG = self.cfg['fft']
            self.MODEL_CFG = self.cfg['model']
            self.FILE_CFG = self.cfg['file']

        self.generator = AudioAutoencoder().to(self.MODEL_CFG['device'])
        self.discriminator = EncodecDiscriminator().to(self.MODEL_CFG['device'])
        
        self.stft_loss = MultiResolutionSTFTLoss()
        
        self.train_loader, self.test_loader = self.getLoader()
        
        self.gen_optim, self.gen_lr_scheduler, self.gen_warmup = self.getOptimizer(
            self.generator, self.MODEL_CFG['lr']
        )
        self.dis_optim, self.dis_lr_scheduler, self.dis_warmup = self.getOptimizer(
            self.discriminator, self.MODEL_CFG['lr']
        )
        
        self.train_losses = {'gen': [], 'dis': []}
        self.eval_losses = []
        self.best_eval_loss = float('inf')
        
        self._setup_wandb()

    def _setup_wandb(self):
        wandb_store = os.path.join(self.FILE_CFG['log_dir'], "wandb_store")
        os.makedirs(wandb_store, exist_ok=True)
        wandb.init(
            project=f"{self.FILE_CFG['project_name']}",
            name=f"{self.FILE_CFG['run_name']}_{datetime.now().strftime('%m%d_%H-%M')}",
            dir=wandb_store,
            config={
                "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected.",
                **self.cfg
            }
        )

    def getOptimizer(self, model, learning_rate):
        warmup_period = self.MODEL_CFG['warmup_period']
        num_steps = len(self.train_loader) * self.MODEL_CFG['epoch'] - warmup_period

        optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period)

        return optimizer, lr_scheduler, warmup_scheduler

    def getLoader(self):
        tensors = load_mp3_files(base_folder=self.FILE_CFG['audio_folder'], config=self.FFT_CFG)
        tensors = torch.cat(tensors, dim=-1)
    
        x = create_overlapping_chunks_tensor(sequence=tensors, config=self.FFT_CFG)
        x = x[torch.randperm(x.size(0))]
    
        train = TensorDataset(x[:-self.FFT_CFG['num_evaluation'], :])
        test = TensorDataset(x[-self.FFT_CFG['num_evaluation']:, :])
        
        train_loader = DataLoader(
            train, 
            batch_size=self.MODEL_CFG['batch_size'], 
            num_workers=self.MODEL_CFG['num_workers'], 
            shuffle=True
        )
        test_loader = DataLoader(
            test, 
            batch_size=self.MODEL_CFG['batch_size'], 
            num_workers=self.MODEL_CFG['num_workers'], 
            shuffle=False
        )
    
        return train_loader, test_loader

    def train_discriminator(self, real_samples, fake_samples):
        self.dis_optim.zero_grad()
        
        dis_real_loss, _, _ = self.discriminator(real_samples, real_samples)
    
        dis_fake_loss, _, _ = self.discriminator(fake_samples.detach(), real_samples)
        
        dis_loss = dis_real_loss + dis_fake_loss
        dis_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.dis_optim.step()
        
        return dis_loss.item()

    def train_generator(self, real_samples, fake_samples):
        self.gen_optim.zero_grad()
        
        _, adv_loss, feature_matching_loss = self.discriminator(fake_samples, real_samples)
        
        stft_loss = self.stft_loss(fake_samples, real_samples)
        
        
        # Total generator loss
        gen_loss = (
            adv_loss + 
            feature_matching_loss + 
            stft_loss
        )
        
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.gen_optim.step()
        
        return gen_loss.item()

    def evaluate(self):
        self.generator.eval()
        eval_losses = []
        
        with torch.no_grad():
            for x in self.test_loader:
                x = x[0].to(self.MODEL_CFG['device'])
                real_samples = x.unsqueeze(1)
                
                latent = self.generator.encode(real_samples)
                fake_samples = self.generator.decode(latent)
                
                stft_loss = self.stft_loss(fake_samples, real_samples)
                
                eval_losses.append(stft_loss.item())
        
        self.generator.train()
        return sum(eval_losses) / len(eval_losses)

    def train(self):
        # Print model information
        total_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        print(f"Generator trainable parameters: {total_params:,}")
        total_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f"Discriminator trainable parameters: {total_params:,}")
        
        EPOCH = self.MODEL_CFG['epoch']
        LOG_DIR = self.FILE_CFG['log_dir']
        os.makedirs(LOG_DIR, exist_ok=True)

        for epoch in range(EPOCH):
            epoch_losses_gen = []
            epoch_losses_dis = []
            
            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCH}")):
                x = x[0].to(self.MODEL_CFG['device'])
                real_samples = x.unsqueeze(1)
                
                # Generate fake samples
                latent = self.generator.encode(real_samples)
                fake_samples = self.generator.decode(latent)
                
                # Train discriminator
                dis_loss = self.train_discriminator(real_samples, fake_samples)
                epoch_losses_dis.append(dis_loss)
                
                # Train generator
                gen_loss = self.train_generator(real_samples, fake_samples)
                epoch_losses_gen.append(gen_loss)
                
                # Warm up leearning rate
                if self.gen_warmup.last_step + 1 >= self.MODEL_CFG['warmup_period']:
                    self.gen_lr_scheduler.step()
                    self.dis_lr_scheduler.step()
                
                wandb.log({
                    "train/gen_loss": gen_loss,
                    "train/dis_loss": dis_loss,
                    "lr": self.gen_optim.param_groups[0]['lr']
                })
            
            avg_gen_loss = sum(epoch_losses_gen) / len(epoch_losses_gen)
            avg_dis_loss = sum(epoch_losses_dis) / len(epoch_losses_dis)
            self.train_losses['gen'].append(avg_gen_loss)
            self.train_losses['dis'].append(avg_dis_loss)
            
            wandb.log({
                "train/epoch_gen_loss": avg_gen_loss,
                "train/epoch_dis_loss": avg_dis_loss
            })
            
            print(f"Epoch {epoch+1}/{EPOCH}")
            print(f"Average Generator Loss: {avg_gen_loss:.4f}")
            print(f"Average Discriminator Loss: {avg_dis_loss:.4f}")
            
            if (epoch + 1) % self.MODEL_CFG['evaluation_cycle'] == 0:
                eval_loss = self.evaluate()
                self.eval_losses.append(eval_loss)
                wandb.log({"eval/loss": eval_loss})
                print(f"Evaluation Loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    torch.save({
                        'generator_state_dict': self.generator.state_dict(),
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'gen_optimizer_state_dict': self.gen_optim.state_dict(),
                        'dis_optimizer_state_dict': self.dis_optim.state_dict(),
                        'epoch': epoch,
                        'best_eval_loss': self.best_eval_loss
                    }, os.path.join(LOG_DIR, 'best_model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()
    
    trainer = Trainer(args.cfg_path)
    trainer.train()