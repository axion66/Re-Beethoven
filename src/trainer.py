import os
import yaml
import torch
import torch.nn as nn
from layers.main_model import net
from layers.diffusion.karrasDiffusion_main import Denoiser
from torch.optim import Adam
from preprocess import load_mp3_files,create_overlapping_chunks_tensor
from torch.utils.data import TensorDataset,DataLoader
import wandb
from datetime import datetime

class Trainer:
    def __init__(self,cfg_path:str):
        self.cfg = self.get_config(cfg_path)
        self.model_cfg = self.cfg['model_param']
        self.net = net(self.model_cfg)
        self.model = Denoiser(model=self.net,sigma_data=0.5)
        self.model_optimizer = Adam(self.model.parameters(),lr=self.model_cfg['lr'],betas=(0.9,0.95),weight_decay=0.1)

    def get_config(self,cfg_path):
        with open(cfg_path) as stream:
            try:
                cfg =yaml.safe_load(stream)
                return cfg
            except yaml.YAMLError as exc:
                raise Exception(f"Incorrect yaml file: {exc}")


    def train(self):
        EPOCH = self.model_cfg['epoch']
        BATCH_SIZE = self.model_cfg['batch_size']
        LOG_FILE = self.model_cfg['log_path']

        train_loader,test_loader = self.get_loader()

        for epoch in range(EPOCH):

            for x
    def get_loader(self):
        tensors,sr = load_mp3_files("../dataset")
        tensors = torch.cat(tensors,dim=-1)
        
        ck_len = self.model_cfg['chunk_len'] # if sr = 8000 and 10 sec, then 80000.
        x = create_overlapping_chunks_tensor(tensors,chunk_len=ck_len)
  

        indices = torch.randperm(x.size(0))
        x = x[indices]

        dSet = {
            'x': x[:x.size()-30,:],
            'x_test': x[x.size(0)-30:,:], # for later evaluation on Discriminator?
        }
        

        BATCH_SIZE = self.model_cfg['batch_size']
        trainDataset,testDataset = TensorDataset(dSet['x']),TensorDataset(dSet['x_test'])
        dLoader,dLoader_test = DataLoader(trainDataset,batch_size=BATCH_SIZE,shuffle=True),DataLoader(testDataset,batch_size=BATCH_SIZE,shuffle=False)

        return dLoader,dLoader_test
        