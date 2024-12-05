import os
import yaml
import torch
import torch.nn as nn
from layers.diffusion.karras import Denoiser
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
        
        self.best_eval_loss = 1e+7
        
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
        
        EPOCH = self.MODEL_CFG['epoch']
        LOG_DIR = self.FILE_CFG['log_dir']
        os.makedirs(LOG_DIR, exist_ok=True)

        step = 0
        scaler = torch.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(self.MODEL_CFG['epoch']):
            
            EPOCH_LOSS = []

            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCH}")):
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
                
                EPOCH_LOSS.append(loss.item())
                wandb.log({"timestamp_loss": loss.item(), "lr": self.optim.param_groups[0]['lr']})
                step += 1
                if step % self.MODEL_CFG['evaluation_cycle'] == 0:
                    self.net.eval()
                    EVAL_LOSS = []
                    with torch.no_grad():
                        for x in self.test_loader:
                            x = x[0].to(self.MODEL_CFG['device']) # list to TEnsor
                            loss = self.denoiser.loss_fn(x)
                            EVAL_LOSS.append(loss.item())
                
                    avg_eval_loss = sum(EVAL_LOSS) / len(EVAL_LOSS)
                    wandb.log({"eval_loss": avg_eval_loss})
                    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    
                    if avg_eval_loss < self.best_eval_loss:
                        self.best_eval_loss = avg_eval_loss
                        torch.save(self.denoiser.state_dict(), os.path.join(LOG_DIR, 'best_model.pth'))
    
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
                    self.net.train()
                
            # Epoch Loss
            avg_train_loss = sum(EPOCH_LOSS) / len(EPOCH_LOSS)
            wandb.log({"train_loss": avg_train_loss})
            print(f"Epoch {epoch+1}/{EPOCH}, Average Train Loss: {avg_train_loss:.4f}")



    
        

    def generate_samples(
            self,
            num_samples: int,
            num_steps: int,
            output_dir: str,
            idx: int
        ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            generated_samples = self.model.sample(num_samples,num_steps)

        for i in range(num_samples):
            sample = generated_samples[i].cpu().numpy().flatten()
            filename = os.path.join(output_dir, f"Sample_{idx}.wav")
            sf.write(filename, sample, self.FFT_CFG['sr'])

        print(f"Generated {num_samples} samples and saved to {output_dir}")    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    trainer = Trainer(args.cfg_path)
    trainer.train()