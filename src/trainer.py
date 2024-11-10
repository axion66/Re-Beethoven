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
import matplotlib.pyplot as plt
import soundfile as sf  # For saving audio files
from tqdm import tqdm
import argparse
#from layers.core_cnn_STFT import net
from layers.cores.core_raw import net
#from layers.core_WavTokenizer import net
#from layers.core_UNet import UNetWithMHA
import pytorch_warmup as warmup


class Trainer:
    def __init__(self, cfg_path: str):
        
        '''CONFIG'''
        with open(cfg_path) as stream:
            self.cfg = yaml.safe_load(stream)
            self.FFT_CFG = self.cfg['fft']
            self.MODEL_CFG = self.cfg['model']
            self.FILE_CFG = self.cfg['file']

        '''MODEL'''
        self.net = net(self.FFT_CFG)
        self.model = Denoiser(config=self.FFT_CFG,model=self.net, sigma_data=0.5,device=torch.device(self.MODEL_CFG['device'])).to(self.MODEL_CFG['device'])
        

        '''Loader'''
        self.train_loader, self.test_loader = self.getLoader()
        self.optim, self.lr_schedule,self.warmup_schedule = self.getOptimizer(model=self.net, trainLoader=self.train_loader, config=self.MODEL_CFG)
        
        '''LOG'''
        self.train_losses = []
        self.eval_losses = []
        self.best_eval_loss = float('inf')

        

        wandb.init(
            project="Audio Diffusion",
            config={
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available.",
                **self.cfg
            },
            name = f"run_{datetime.now().strftime('%m%d_%H-%M')}"
        )
        

    def getOptimizer(self, model, trainLoader, config):
        warmup_period = config['warmup_period']
        num_steps = len(trainLoader) * config['epoch'] - warmup_period

        optimizer = Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period)

        return optimizer,lr_scheduler,warmup_scheduler
 
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
                self.optim.zero_grad()
                x = x[0].to(self.MODEL_CFG['device'])
                loss = self.model.loss_fn(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optim.step()
                with self.warmup_schedule.dampening():
                    if self.warmup_schedule.last_step + 1 >= self.MODEL_CFG['warmup_period']:
                        self.lr_schedule.step()
                
                # Timestamp Loss, LR
                EPOCH_LOSS.append(loss.item())
                wandb.log({"timestamp_loss": loss.item(), "lr": self.optim.param_groups[0]['lr']})

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
                    for x in self.test_loader:
                        x = x[0].to(self.MODEL_CFG['device']) # list to TEnsor
                        loss = self.model.loss_fn(x)
                        EVAL_LOSS.append(loss.item())
            
                avg_eval_loss = sum(EVAL_LOSS) / len(EVAL_LOSS)
                self.eval_losses.append(avg_eval_loss)
                wandb.log({"eval_loss": avg_eval_loss})
                print(f"Evaluation Loss: {avg_eval_loss:.4f}")

                if avg_eval_loss < self.best_eval_loss:
                    self.best_eval_loss = avg_eval_loss
                    torch.save(self.model.state_dict(), os.path.join(LOG_DIR, 'best_model.pth'))

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