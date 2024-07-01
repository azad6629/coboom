import torch
from torchvision import datasets
from .byol_transform import get_transform, MultiViewDataInjector

import numpy as np
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import cv2
import os
import torchvision.transforms as transforms
from pathlib import Path


class NIHdataset(torch.utils.data.Dataset):
    def __init__(self, image_filepaths, transform):
        self.image_filepaths = image_filepaths
        self.transform = transform

    def __getitem__(self, idx):
    
        img = self.image_filepaths[idx]
        image = Image.open(img).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(1)
    
        return image,label

    def __len__(self):
        return len(self.image_filepaths)


class NihDataLoader():
    def __init__(self, config):
        self.config = config
        self.resize_size = config['data']['resize_size']
        
        self.train_df  = pd.read_csv(config['data']['train_df']) 
        self.valid_df  = pd.read_csv(config['data']['valid_df']) 
        
        self.data_workers = config['data']['data_workers']
        self.views = config['data']['dual_views']
        
    def getnihchex14_dataset(self):
        image_filepaths = self.train_df["filename"].values        
        image_size = self.resize_size
        transform = get_transform(image_size)
              
        if self.views ==2:
            transform  = MultiViewDataInjector([transform,transform])
                    
        dataset = NIHdataset(image_filepaths,transform=transform)
        print(f'{len(dataset)} images have loaded')
        print('dataset length :', len(dataset))
        return dataset
            
    def NIH_loader(self, batch_size):
        dataset = self.getnihchex14_dataset()
        self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle= True,
                                                  num_workers=self.data_workers,
                                                  pin_memory=True,
                                                  sampler=self.train_sampler,
                                                  drop_last=True)
        return data_loader
    
    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)