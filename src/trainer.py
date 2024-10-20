import os
import yaml
import torch
import torch.nn as nn
from layers.main_model import net
from layers.diffusion.karrasDiffusion_main import Denoiser,KarrasSampler,KarrasSchedule
from torch.optim import Adam
from layers.preprocess import load_mp3_files, create_overlapping_chunks_tensor
from torch.utils.data import TensorDataset, DataLoader
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import soundfile as sf  # For saving audio files
from tqdm import tqdm
import argparse



class Trainer:
    def __init__(self, cfg_path: str):
        self.cfg = self.get_config(cfg_path)
        self.fft_setup = self.cfg['fft_setup']
        self.model_param = self.cfg['model_param']
        self.file_path = self.cfg['file_path']

        self.net = net(self.fft_setup)
        self.model = Denoiser(model=self.net, sigma_data=0.5).to(self.model_param['device'])
        self.model_optimizer = Adam(self.model.parameters(), lr=self.model_param['lr'], betas=(0.9, 0.95), weight_decay=0.1)

        self.train_losses = []
        self.eval_losses = []
        self.best_eval_loss = float('inf')

        self.train_loader, self.test_loader = self.get_loader()

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project="audio-gen",
            config={
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available.",
                **self.cfg
            },
            name=f"run_{current_time}"
        )


        # Initialize sampler and scheduler
        self.sampler = KarrasSampler()
        self.scheduler = KarrasSchedule()

    def get_config(self, cfg_path):
        with open(cfg_path) as stream:
            try:
                cfg = yaml.safe_load(stream)
                return cfg
            except yaml.YAMLError as exc:
                raise Exception(f"Incorrect yaml file: {exc}")

    def train(self):
        EPOCH = self.model_param['epoch']
        BATCH_SIZE = self.model_param['batch_size']
        LOG_DIR = self.file_path['log_dir']
        os.makedirs(LOG_DIR, exist_ok=True)

        for epoch in range(self.model_param['epoch']):
            self.model.train()
            epoch_losses = []

            for i, x in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCH}")):
                self.model_optimizer.zero_grad()
                x = x[0].to(self.model_param['device'])
                x_denoised, sigmas = self.model(x)
                loss = self.model.calculate_loss(x, x_denoised, sigmas)
                loss.backward()

                self.model_optimizer.step()

                epoch_losses.append(loss.item())

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            self.train_losses.append(avg_train_loss)

            # Log training loss to wandb
            wandb.log({"train_loss": avg_train_loss})

            print(f"Epoch {epoch+1}/{EPOCH}, Average Train Loss: {avg_train_loss:.4f}")

            # Evaluate the model and generate samples every 20 epochs
            if (epoch + 1) % 20 == 0:
                eval_loss = self.evaluate(self.test_loader)
                self.eval_losses.append(eval_loss)
                wandb.log({"eval_loss": eval_loss})
                print(f"Evaluation Loss: {eval_loss:.4f}")

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    torch.save(self.model.state_dict(), os.path.join(LOG_DIR, 'best_model.pth'))

                # Generate samples
                num_samples = 5  # You can adjust this number
                num_steps = 1000  # You can adjust this number
                samples_dir = os.path.join(LOG_DIR, f"samples_epoch_{epoch+1}")
                self.generate_samples(num_samples, num_steps, samples_dir)

                # Log sample audio to wandb
                for i in range(num_samples):
                    sample_path = os.path.join(samples_dir, f"sample_{i}.wav")
                    wandb.log({f"audio_sample_{i}": wandb.Audio(sample_path, sample_rate=self.fft_setup['sr'])})

            # Plot and save loss curves
            self.plot_losses(LOG_DIR)


    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x[0].to(self.model_param['device']) # list to TEnsor
                x_denoised, sigmas = self.model(x)
                loss = self.model.calculate_loss(x, x_denoised, sigmas)
                total_loss += loss.item()
        return total_loss / len(test_loader)

    def plot_losses(self, log_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', color='blue')

        if self.eval_losses:  # Check if there are any eval losses
            eval_x = range(0, len(self.eval_losses) * 20, 20)
            plt.plot(eval_x, self.eval_losses, label='Eval Loss', color='orange', marker='o')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Losses')
        plt.legend()
        
        plt.savefig(os.path.join(log_dir, 'loss_plot.png'))
        plt.close()

    def get_loader(self):
        tensors = load_mp3_files(self.file_path['audio_folder'],self.fft_setup)
        tensors = torch.cat(tensors, dim=-1)

        x = create_overlapping_chunks_tensor(tensors, self.fft_setup)

        indices = torch.randperm(x.size(0))
        x = x[indices]

        dSet = {
            'x': x[:-self.fft_setup['num_test_samples'], :],
            'x_test': x[-self.fft_setup['num_test_samples']:, :],
        }

        BATCH_SIZE = self.model_param['batch_size']
        trainDataset, testDataset = TensorDataset(dSet['x']), TensorDataset(dSet['x_test'])
        dLoader, dLoader_test = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)

        return dLoader, dLoader_test


    def generate_samples(self, num_samples: int, num_steps: int, output_dir: str) -> None:
        audio_path = output_dir
        os.makedirs(audio_path, exist_ok=True)

        noise = torch.randn(num_samples, self.fft_setup['seq_len']).to(self.model_param['device']) 
        sigmas = self.scheduler(num_steps=num_steps, device=self.model_param['device'])
   
        with torch.no_grad():
            generated_samples = self.sampler(noise, model=self.model, sigmas=sigmas, num_steps=num_steps)

        for i in range(num_samples):
            sample = generated_samples[i].cpu().numpy().flatten()
            filename = os.path.join(audio_path, f"sample_{i}.wav")
            sf.write(filename, sample, self.fft_setup['sr'])

        print(f"Generated {num_samples} samples and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="Path to the configuration YAML file")
    
    args = parser.parse_args()
    
    trainer = Trainer(args.cfg_path)
    trainer.train()