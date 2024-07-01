# -*- coding:utf-8 -*-
# +
import torch
from torchvision import datasets
from .byol_transform_strong import get_transform_strong, MultiViewDataInjector
from .byol_transform_weak import get_transform_weak


import numpy as np
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import cv2
from transformers import AutoTokenizer, AutoModel
import os
import torchvision.transforms as transforms
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -

from pathlib import Path
import yaml


# IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
# IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)
def get_vae_transform():
    mean = [0.4760, 0.4760, 0.4760]
    std  = [0.3001, 0.3001, 0.3001]
    imgtransCrop = 224

    # Tranform data
    normalize = transforms.Normalize(mean, std)
    transformList = []
    transformList.append(transforms.Resize((imgtransCrop, imgtransCrop)))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transform = transforms.Compose(transformList)
    
    return transform


# +
class mimicdataset(torch.utils.data.Dataset):
    def __init__(self, exp_mode,df,rad_df,class_names, transform,policy="ones"): #captions, tokenizer
        self.pathologies = class_names
        self.pathologies = sorted(self.pathologies)
        self.data_df = df
        self.rad_df = rad_df
        self.transform = transform
        self.vae_transform = get_vae_transform()
#         self.captions = list(captions)
        self.exp_mode = exp_mode
#         self.encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=510)
        
        
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.data_df.columns:
                mask = self.data_df[pathology]
            self.labels.append(mask.values)
            
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        if policy == "ones":
            self.labels[self.labels == -1] = 1
        elif policy == "zeroes":
            self.labels[self.labels == -1]= 0
        else:
            self.labels[self.labels == -1] = np.nan

    def __getitem__(self, idx):
#         item = {
#             key: torch.tensor(values[idx])
#             for key, values in self.encoded_captions.items()  #.detach().clone()
#         }
        img_path = self.data_df['path'][idx]
        image = Image.open(img_path).convert('RGB')
    
        if self.vae_transform:
            vae_image = self.vae_transform(image)
        
        if self.transform:
            image = self.transform(image)
               
        rad_path = self.rad_df['Path'][idx]  
#         assert (img_path == rad_path)
        rad_features = self.rad_df.iloc[[idx],1:].values
#         print(rad_features)
            
#         item['image'] = image    
#         item['caption'] = self.captions[idx]
#         item['rad_features'] = rad_features
        
        
        label = self.labels[idx]
        
        if self.exp_mode == 'bert_byol':
            return item['image'],item["input_ids"], item["attention_mask"]
        elif self.exp_mode == 'rad_bert_byol':
            return item['image'],item["input_ids"], item["attention_mask"],item['rad_features']
        elif self.exp_mode == 'rad_byol':
            return item['image'],item['rad_features']
        elif self.exp_mode == 'byol':
            return image,label
        elif self.exp_mode == 'vae_byol':
            return image,vae_image
            

    def __len__(self):
        return len(self.data_df)


# -

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


class ChestX_ray14(torch.utils.data.Dataset):

    def __init__(self, pathImageDirectory, pathDatasetFile, augment):

        self.img_list = []
        self.augment = augment
        self.labels = []
        self.vae_transform = get_vae_transform()


        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()
                    
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    self.img_list.append(imagePath)
                    label = lineItems[1:]
                    self.labels.append(label)
        
        self.labels = np.asarray(self.labels)
        self.labels = self.labels.astype(np.float32)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        labels = self.labels[index]
        if self.vae_transform:
            vae_image = self.vae_transform(imageData)
        
        return self.augment(imageData),vae_image

    def __len__(self):
        return len(self.img_list)


# +
class ImageNetLoader():
    def __init__(self, config):
        self.config = config
        self.resize_size = config['data']['resize_size']
        
        self.train_df  = pd.read_csv(config['data']['train_df']) 
        self.valid_df  = pd.read_csv(config['data']['valid_df']) 

        self.nih_train_list = config['data']['nih_train_list']  
        self.nih_val_list = config['data']['nih_val_list']   
        
        self.data_workers = config['data']['data_workers']
        self.views = config['data']['dual_views']
        self.exp_mode = config['exp_mode']
        self.aug_mode = config['aug_mode']
        
    def getnihchex14_dataset(self):
        image_filepaths = self.train_df["filename"].values        
        image_size = self.resize_size
        if self.aug_mode == 'strong':
            transform = get_transform_strong(image_size)
        if self.aug_mode == 'week':
            transform = get_transform_weak(image_size)

              
        if self.views ==2:
            transform  = MultiViewDataInjector([transform,transform])
                    
        dataset = NIHdataset(image_filepaths,transform=transform)
        print(f'{len(dataset)} images have loaded')
        print('dataset length :', len(dataset))
        return dataset
        
        
    def getnih_dataset(self):
        train_list = self.nih_train_list
        image_size = self.resize_size
        
        if self.aug_mode == 'strong':
            transform_s = get_transform_strong(image_size)
            transform_w = get_transform_weak(image_size)
        if self.aug_mode == 'week':
            transform = get_transform_weak(image_size)

              
        if self.views ==2:
            transform  = MultiViewDataInjector([transform_w,transform_s])
                
        train_set = ChestX_ray14(pathImageDirectory = '/workspace/data/DATASETS/NIH_Chest-Xray-14/images/',
                                     pathDatasetFile=train_list,
                                     augment=transform)
        
        print(f'{len(train_set)} images have loaded')
        print('dataset length :', len(train_set))
        return train_set
    
    def getmimic_dataset(self):
        train_df = self.train_df
        rad_df   = self.rad_df
        
#         captions = self.train_df["report"].values
#         tokenizer = AutoTokenizer.from_pretrained(self.config['text_tokenizer'])
        
        class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        
        image_size = self.resize_size
        if self.aug_mode == 'strong':
            transform = get_transform_strong(image_size)
        if self.aug_mode == 'week':
            transform = get_transform_weak(image_size)

              
        if self.views ==2:
            transform  = MultiViewDataInjector([transform,transform])
        
            
        dataset = mimicdataset(self.exp_mode,
                               train_df,
                               rad_df,
                               class_names,
#                                captions,
#                                tokenizer,
                               transform=transform)
        
        print(f'{len(dataset)} images have loaded')
        print('dataset length :', len(dataset))
        return dataset
    
    def mimic_loader(self, batch_size):
        dataset = self.getmimic_dataset()
        self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle= True,
                                                  num_workers=self.data_workers,
                                                  pin_memory=True,
                                                  sampler=self.train_sampler,
                                                  drop_last=True
                                            )
        return data_loader
    
    def NIH_loader(self, batch_size):
        dataset = self.getnihchex14_dataset()
#         dataset = self.getnih_dataset()
        self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle= True,
                                                  num_workers=self.data_workers,
                                                  pin_memory=True,
                                                  sampler=self.train_sampler,
                                                  drop_last=True
                                            )
        return data_loader
    
     
    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)


# +
# with open('/workspace/mayankk/Radiomics/BERT-BYOL/config/train_config_mimic_2T.yaml', 'r') as f:
#     config = yaml.safe_load(f)

# data_ins = ImageNetLoader(config)

# train_loader = data_ins.mimic_loader(1)

# for idn, (img,lbl) in enumerate(train_loader):
#     print(idn,img.shape)
# #     if idx ==0:
# #         break
# -


