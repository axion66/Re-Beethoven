import os
import yaml
import torch
import torch.nn as nn
from layers.gan.simple import GANWrapper
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
#from layers.cores.core_raw import net
#from layers.core_WavTokenizer import net
import pytorch_warmup as warmup
from layers.cores.unet import net

class Trainer:
    def __init__(self, cfg_path: str):
        '''CONFIG'''
        with open(cfg_path) as stream:
            self.cfg = yaml.safe_load(stream)
            self.FFT_CFG = self.cfg['fft']
            self.MODEL_CFG = self.cfg['model']
            self.FILE_CFG = self.cfg['file']

        '''MODEL'''
        self.gen = net(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            num_res_blocks=2,
            attention_resolutions=[2],
            dropout=0.1,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=4,
            use_scale_shift_norm=False,
            resblock_updown=True
        )
        self.dis = net(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=[],
            dropout=0.1,
            channel_mult=(1, 2),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=4,
            use_scale_shift_norm=False,
            resblock_updown=True,
            gan_mode=True
        )

        print("Model prepared.")
        self.device = torch.device(self.MODEL_CFG['device'])
        self.model = GANWrapper(self.FFT_CFG, generator=self.gen, discriminator=self.dis).to(self.device)

        '''Loader'''
        self.train_loader, self.test_loader = self.getLoader()
        self.optim_gen = self.getOptimizer(model=self.model.generator, trainLoader=self.train_loader, config=self.MODEL_CFG)
        self.optim_dis = self.getOptimizer(model=self.model.discriminator, trainLoader=self.train_loader, config=self.MODEL_CFG)
        
        '''LOG'''
        self.train_losses = []
        self.eval_losses = []
        self.best_eval_loss = float('inf')

        wandb.init(
            project="Audio Diffusion(GAN)",
            config={
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available.",
                **self.cfg
            },
            name=f"run_{datetime.now().strftime('%m%d_%H-%M')}"
        )

    def getOptimizer(self, model, trainLoader, config):
        optimizer = Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.1)
        return optimizer
 
    def train(self):
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}") 
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)  
        print(f"Total trainable parameters: {total_params:,}")
        
        EPOCH = self.MODEL_CFG['epoch']
        LOG_DIR = self.FILE_CFG['log_dir']
        os.makedirs(LOG_DIR, exist_ok=True)

        # Add sigmoid for discriminator output
        sigmoid = torch.nn.Sigmoid()

        for epoch in range(EPOCH):
            self.model.train()
            EPOCH_LOSS = []

            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCH}")):
                try:
                    x = x[0].to(self.device)
                    batch_size = x.size(0)
                    
                    # Normalize input with clamping to prevent extreme values
                    x_mean, x_std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
                    x = torch.clamp((x - x_mean) / (x_std + 1e-8), -10, 10)
                    
                    # Train Generator
                    self.optim_gen.zero_grad()
                    noise = torch.randn(batch_size, x.size(1), device=self.device)
                    condition = torch.zeros((batch_size, 1), device=self.device)
                    
                    fake_samples = self.model.generator(noise, condition)
                    dis_fake = self.model.discriminator(fake_samples, condition)
                    
                    # Apply sigmoid to discriminator output
                    dis_fake = sigmoid(dis_fake)
                    
                    # Create labels with noise for label smoothing
                    real_labels = torch.ones_like(dis_fake, device=self.device) * 0.9  # Label smoothing
                    fake_labels = torch.zeros_like(dis_fake, device=self.device) + 0.1  # Label smoothing
                    
                    # Calculate generator loss
                    gen_loss = self.model.advLoss(dis_fake, real_labels)
                    
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=1.0)
                    gen_loss.backward()
                    self.optim_gen.step()

                    # Train Discriminator
                    self.optim_dis.zero_grad()
                    
                    # Real samples
                    dis_real = sigmoid(self.model.discriminator(x, condition))
                    real_loss = self.model.advLoss(dis_real, real_labels)
                    
                    # Fake samples
                    dis_fake = sigmoid(self.model.discriminator(fake_samples.detach(), condition))
                    fake_loss = self.model.advLoss(dis_fake, fake_labels)
                    
                    # Combined discriminator loss
                    dis_loss = (real_loss + fake_loss) * 0.5
                    
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                    dis_loss.backward()
                    self.optim_dis.step()

                    # Debug prints
                    if i % 100 == 0:  # Print every 100 batches
                        print(f"\nDebug info for batch {i}:")
                        print(f"Discriminator real output range: [{dis_real.min().item():.4f}, {dis_real.max().item():.4f}]")
                        print(f"Discriminator fake output range: [{dis_fake.min().item():.4f}, {dis_fake.max().item():.4f}]")
                        print(f"Generator loss: {gen_loss.item():.4f}")
                        print(f"Discriminator loss: {dis_loss.item():.4f}")

                    EPOCH_LOSS.append(gen_loss.item())
                    
                    # Logging
                    wandb.log({
                        "generator_loss": gen_loss.item(),
                        "discriminator_loss": dis_loss.item(),
                        "discriminator_real_loss": real_loss.item(),
                        "discriminator_fake_loss": fake_loss.item(),
                        "dis_real_max": dis_real.max().item(),
                        "dis_real_min": dis_real.min().item(),
                        "dis_fake_max": dis_fake.max().item(),
                        "dis_fake_min": dis_fake.min().item()
                    })

                except RuntimeError as e:
                    print(f"Error in batch {i}: {str(e)}")
                    print(f"Current tensor states:")
                    print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
                    continue

            # Epoch Loss
            if EPOCH_LOSS:  # Check if we have any valid losses
                avg_train_loss = sum(EPOCH_LOSS) / len(EPOCH_LOSS)
                self.train_losses.append(avg_train_loss)
                wandb.log({"train_loss": avg_train_loss})
                print(f"Epoch {epoch+1}/{EPOCH}, Average Train Loss: {avg_train_loss:.4f}")

            # Evaluation & Sampling
            if (epoch + 1) % self.MODEL_CFG['evaluation_cycle'] == 0:
                self.evaluate(epoch, LOG_DIR)
    def evaluate(self, epoch, LOG_DIR):
        self.model.eval()
        EVAL_LOSS = []
        
        with torch.no_grad():
            for x in self.test_loader:
                x = x[0].to(self.device)
                batch_size = x.size(0)
                sigmoid = torch.nn.Sigmoid()
                # Normalize input
                x_mean, x_std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
                x = (x - x_mean) / (x_std + 1e-8)
                
                condition = torch.zeros((batch_size, 1), device=self.device)
                noise = torch.randn(batch_size, x.size(1), device=self.device)
                
                fake_samples = self.model.generator(noise, condition)
                
                dis_real = sigmoid(self.model.discriminator(x, condition))
                dis_fake = sigmoid(self.model.discriminator(fake_samples, condition))
                
                real_loss = self.model.advLoss(dis_real, torch.ones_like(dis_real, device=self.device))
                fake_loss = self.model.advLoss(dis_fake, torch.zeros_like(dis_fake, device=self.device))
                
                eval_loss = (real_loss + fake_loss) * 0.5
                EVAL_LOSS.append(eval_loss.item())

        avg_eval_loss = sum(EVAL_LOSS) / len(EVAL_LOSS)
        self.eval_losses.append(avg_eval_loss)
        wandb.log({"eval_loss": avg_eval_loss})
        print(f"Evaluation Loss: {avg_eval_loss:.4f}")

        if avg_eval_loss < self.best_eval_loss:
            self.best_eval_loss = avg_eval_loss
            torch.save(self.model.state_dict(), os.path.join(LOG_DIR, 'best_model.pth'))

        # Generate samples
        STORE_SAMPLE_DIR = os.path.join(LOG_DIR, f"samples_epoch_{epoch+1}")
        os.makedirs(STORE_SAMPLE_DIR, exist_ok=True)
        
        for idx in range(self.MODEL_CFG['num_samples']):
            self.generate_samples(
                num_samples=1,
                num_steps=self.MODEL_CFG['sampling_steps'][idx],
                output_dir=STORE_SAMPLE_DIR,
                idx=idx
            )
            
            sample_path = os.path.join(STORE_SAMPLE_DIR, f"Sample_{idx}.wav")
            wandb.log({f"Sample {idx}": wandb.Audio(sample_path, sample_rate=self.FFT_CFG['sr'])})


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
            generated_samples = self.model.sample(num_samples)

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