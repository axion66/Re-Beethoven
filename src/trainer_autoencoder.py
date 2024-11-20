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
import numpy as np
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
        
        dis_loss, real_loss, fake_loss = self.discriminator.loss(real_samples, fake_samples)
        dis_loss.backward()
        self.dis_optim.step()
        
        # Log individual components of discriminator loss
        wandb.log({
            "train/discriminator/dis_loss": dis_loss.item(),
        })
        
        return dis_loss.item()

    def train_generator(self, real_samples, fake_samples, info):
        self.gen_optim.zero_grad()
        
        dis_loss, adv_loss, feature_matching_loss = self.discriminator.loss(real_samples, fake_samples)
        stft_loss = self.stft_loss(fake_samples, real_samples)
        
        gen_loss = adv_loss + feature_matching_loss + stft_loss + 1e-4 * info['kl']
        
        gen_loss.backward()
        self.gen_optim.step()
        
        # Log individual components of generator loss
        wandb.log({
            "train/generator/gen_loss": gen_loss.item(),
            "train/generator/adv_loss": adv_loss.item(),
            "train/generator/feature_matching_loss": feature_matching_loss.item(),
            "train/generator/stft_loss": stft_loss.item(),
            "train/generator/kl_loss": (1e-4 * info['kl']).item()
        })
        
        return gen_loss.item()  

    def sample_and_save_audio(self, real_samples, fake_samples, epoch):
        epoch_dir = os.path.join(self.FILE_CFG['log_dir'], f'samples/epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        num_samples = min(5, real_samples.size(0))
        
        for i in range(num_samples):
            real_path = os.path.join(epoch_dir, f'real_sample_{i}.wav')
            fake_path = os.path.join(epoch_dir, f'generated_sample_{i}.wav')
            self.save_audio_sample(real_samples[i], real_path)
            self.save_audio_sample(fake_samples[i], fake_path)
            
            wandb.log({
                f"audio_samples/real_{i}": wandb.Audio(real_path, sample_rate=8000),
                f"audio_samples/generated_{i}": wandb.Audio(fake_path, sample_rate=8000)
            })

    def save_audio_sample(self, audio_tensor, filename, sample_rate=8000):
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.squeeze(0)
        audio_np = audio_tensor.cpu().numpy().astype(np.float32)
        audio_np = np.clip(audio_np, -1, 1)
        sf.write(filename, audio_np, sample_rate)

    def evaluate(self, epoch):
        self.generator.eval()
        eval_losses = []
        sample_real, sample_fake = None, None
        
        with torch.no_grad():
            for x in self.test_loader:
                x = x[0].to(self.MODEL_CFG['device'])
                real_samples = x.unsqueeze(1)
                
                latent, info = self.generator.encode(real_samples, return_info=True)
                fake_samples = self.generator.decode(latent)
                
                stft_loss = self.stft_loss(fake_samples, real_samples)
                
                eval_losses.append(stft_loss.item())
                if sample_real is None:
                    sample_real = real_samples.squeeze(1)
                    sample_fake = fake_samples.squeeze(1)
        
        if sample_real is not None and sample_fake is not None:
            self.sample_and_save_audio(sample_real, sample_fake, epoch)
        
        self.generator.train()
        return sum(eval_losses) / len(eval_losses)

    def train(self):
        total_params_gen = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        total_params_dis = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f"Generator trainable parameters: {total_params_gen:,}")
        print(f"Discriminator trainable parameters: {total_params_dis:,}")
        
        for epoch in range(self.MODEL_CFG['epoch']):
            epoch_losses_gen, epoch_losses_dis = [], []
            
            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
                x = x[0].to(self.MODEL_CFG['device'])
                real_samples = x.unsqueeze(1)
                latent, info = self.generator.encode(real_samples, return_info=True)
                fake_samples = self.generator.decode(latent)

                if i % 2 == 0:
                    dis_loss = self.train_discriminator(real_samples, fake_samples)
                    epoch_losses_dis.append(dis_loss)
                    wandb.log({"train/dis_loss": dis_loss})
                else:
                    gen_loss = self.train_generator(real_samples, fake_samples, info)
                    epoch_losses_gen.append(gen_loss)
                    wandb.log({"train/gen_loss": gen_loss})

                self.gen_lr_scheduler.step()
                self.dis_lr_scheduler.step()
                wandb.log({"lr": self.gen_optim.param_groups[0]['lr']})
            
            avg_gen_loss = sum(epoch_losses_gen) / max(1, len(epoch_losses_gen))
            avg_dis_loss = sum(epoch_losses_dis) / max(1, len(epoch_losses_dis))
            
            self.train_losses['gen'].append(avg_gen_loss)
            self.train_losses['dis'].append(avg_dis_loss)
            
            wandb.log({
                "train/epoch_gen_loss": avg_gen_loss,
                "train/epoch_dis_loss": avg_dis_loss
            })
            
            if (epoch + 1) % self.MODEL_CFG['evaluation_cycle'] == 0:
                eval_loss = self.evaluate(epoch)
                self.eval_losses.append(eval_loss)
                wandb.log({"eval/loss": eval_loss})
                
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    torch.save({
                        'generator_state_dict': self.generator.state_dict(),
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'gen_optimizer_state_dict': self.gen_optim.state_dict(),
                        'dis_optimizer_state_dict': self.dis_optim.state_dict(),
                        'epoch': epoch,
                        'best_eval_loss': self.best_eval_loss
                    }, os.path.join(self.FILE_CFG['log_dir'], 'best_model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()
    
    trainer = Trainer(args.cfg_path)
    trainer.train()
