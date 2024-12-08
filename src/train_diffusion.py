import os
import yaml
import torch
import torch.nn as nn
from layers.diffusion.v import Denoiser
from torch.optim import AdamW
from layers.preprocess import load_files, create_overlapping_chunks_tensor
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import soundfile as sf  
from tqdm import tqdm
import wandb
import argparse
import pytorch_warmup as warmup
from layers.cores.dit import DiT
from layers.autoencoder.vae import AutoEncoderWrapper,AudioAutoencoder
from layers.tools.losses import *


def print_model_size(name, model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()}") 
            
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"{name} | Trainable[requires_grad] parameters: {total_params}")
    
    
class Trainer:
    def __init__(self, cfg_path):
        
        self.configure_config(cfg_path = cfg_path)
        self.configure_model()
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
        self.autoencoder = AutoEncoderWrapper(
            autoencoder = AudioAutoencoder(),
            autoencoder_state_path = self.MODEL_CFG['pretrained_autoencoder']
        )
        self.dit = DiT(
            config = self.FFT_CFG,
            pretrained_autoencoder = self.autoencoder
        )
        self.denoiser = Denoiser(
            config = self.FFT_CFG,
            model = self.dit, 
            sigma_data = 0.5,
            device = torch.device(self.MODEL_CFG['device'])
        ).to(self.MODEL_CFG['device'])

        if (self.MODEL_CFG['resume_diffusion']):
            state = torch.load(self.MODEL_CFG['resume_diffusion_path'])
            self.denoiser.load_state_dict(state, strict = True)
            
        print_model_size("Autoencoder", self.autoencoder)
        print_model_size("DiT", self.dit)
        print_model_size("Denoiser", self.denoiser)
    
    def configure_loader(self):
        batches = load_files(base_folder=self.FILE_CFG['audio_folder'], config=self.FFT_CFG)
        tensors = torch.cat(batches, dim=-1)

        x = create_overlapping_chunks_tensor(sequence = tensors, config = self.FFT_CFG)
        x = x[torch.randperm(x.size(0))]

        train = TensorDataset(x[:-self.FFT_CFG['num_evaluation'], :])
        test = TensorDataset(x[-self.FFT_CFG['num_evaluation']:, :])
        
        self.train_loader = DataLoader(
            dataset = train, 
            batch_size = self.MODEL_CFG['batch_size'], 
            num_workers = self.MODEL_CFG['num_workers'], 
            shuffle = True, 
            drop_last = True,
        )
        
        self.test_loader = DataLoader(
            dataset = test, 
            batch_size = self.MODEL_CFG['batch_size'], 
            num_workers = self.MODEL_CFG['num_workers'], 
            shuffle = False,
            drop_last = False
        )

    def configure_optimizer(self):
        warmup_period = self.MODEL_CFG['warmup_period']
        num_steps = len(self.train_loader) * self.MODEL_CFG['epoch'] - warmup_period
        self.optim = AdamW(self.denoiser.parameters(recurse = True), lr = self.MODEL_CFG['lr'], betas = (0.9, 0.999), weight_decay=0.01)
        self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=num_steps)
        self.warmup_schedule = warmup.ExponentialWarmup(self.optim, warmup_period)

    def configure_wandb(self):
        wandb_store = os.path.join(self.FILE_CFG['log_dir'],"wandb_store")
        os.makedirs(wandb_store, exist_ok=True)
        
        wandb.init(
            project = f"{self.FILE_CFG['project_name']}",
            name = f"{self.FILE_CFG['run_name']}_{datetime.now().strftime('%m%d_%H-%M')}",
            dir = wandb_store,
            config={
                "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected.",
                **self.cfg
            }
        )
        
    def train(self):
        
        LOG_DIR = self.FILE_CFG['log_dir']
        os.makedirs(LOG_DIR, exist_ok=True)
        self.SAMPLE_DIR = os.path.join(LOG_DIR, "samples")
        os.makedirs(self.SAMPLE_DIR, exist_ok=True)
        self.best_eval_loss = 1e+7

        step = 0
        scaler = torch.GradScaler(device = "cuda" if torch.cuda.is_available() else "cpu")
        eval_step = self.MODEL_CFG['evaluation_cycle']
        for epoch in range(self.MODEL_CFG['epoch']):
            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.MODEL_CFG['epoch']}")):
                self.optim.zero_grad()

                x = x[0].to(self.MODEL_CFG['device'])
                with torch.autocast(device_type = "cuda" if torch.cuda.is_available() else "cpu", dtype = torch.float16):
                    loss = self.denoiser.loss_fn(x)
                
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()

                with self.warmup_schedule.dampening():
                    if self.warmup_schedule.last_step + 1 >= self.MODEL_CFG['warmup_period']:
                        self.lr_schedule.step()

                wandb.log({"loss/train": loss.item(), "lr": self.optim.param_groups[0]['lr']})
                


                step += 1
                if step % eval_step == 0:
                    with torch.no_grad():
                        sum_loss = 0.0
                        FAKE_DIR = os.path.join(self.SAMPLE_DIR, f"generated_epoch_{epoch+1}")
                        ORIG_DIR = os.path.join(self.SAMPLE_DIR, f"original_epoch_{epoch+1}")
                        os.makedirs(FAKE_DIR, exist_ok = True)
                        os.makedirs(ORIG_DIR, exist_ok = True)
                        for x in self.test_loader:
                            x = x[0].to(self.MODEL_CFG['device']) 
                            with torch.autocast(device_type = "cuda" if torch.cuda.is_available() else "cpu", dtype = torch.float16):
                                loss = self.denoiser.loss_fn(x)
                                decoded, sigmas = self.denoiser(x)
                            wandb.log({"loss/evaluation": loss})
                            sum_loss += loss
                            self.generate_samples(decoded,output_dir=FAKE_DIR, name="denoised")
                            self.generate_samples(x, output_dir=ORIG_DIR, name="real")

                        
                        wandb.log({"loss/evaluation_mean": sum_loss})
                        if sum_loss < self.best_eval_loss:
                            self.best_eval_loss = sum_loss
                            torch.save(self.denoiser.state_dict(), os.path.join(LOG_DIR, 'best_model.pth'))
                            print(f"Best Model saved at epoch {epoch}")
                            
                        self.sample(
                            num_samples = self.MODEL_CFG['num_samples'],
                            epoch = epoch,
                        )

    @torch.no_grad()  
    def generate_samples(
            self,
            x,
            output_dir: str,
            name: str,
        ) -> None:
        x = x.cpu()
        for i in range(x.size(0)):
            sample = x[i].numpy().flatten()
            filename = os.path.join(output_dir, f"{name}_{i}.wav")
            sf.write(filename, sample, self.FFT_CFG['sr'])
            wandb.log({f"{name}/Sample {i}": wandb.Audio(sample, sample_rate=self.FFT_CFG['sr'])})


    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            epoch: int,
        ) -> None:

        STORE_SAMPLE_DIR = os.path.join(self.SAMPLE_DIR, f"Epoch_{epoch+1}")
        os.makedirs(STORE_SAMPLE_DIR, exist_ok=True)

        generated_samples = self.denoiser.sample(num_samples).cpu()

        for i in range(num_samples):
            sample = generated_samples[i].numpy().flatten()
            filename = os.path.join(STORE_SAMPLE_DIR, f"{i}.wav")
            sf.write(filename, sample, self.FFT_CFG['sr'])
            wandb.log({f"samples/{i}": wandb.Audio(sample, sample_rate=self.FFT_CFG['sr'])})
            
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, default="configs/config.yml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    trainer = Trainer(args.cfg_path)
    trainer.train()