# -*- coding:utf-8 -*-
# +
#plain encoder and vae projection... no codebook
# -

import time
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from .v2_vq_vae import model as vae

from model import VAE_BYOLModel
from optimizer import LARS
from data import ImageNetLoader
from utils import params_util, logging_util, eval_util
from utils.data_prefetcher import data_prefetcher
# from utils.data_prefetcher_bert import data_prefetcher
from utils.logging_util import log
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VAE_Projection(nn.Module):
    def __init__(self):
        super(VAE_Projection, self).__init__()
        # Convolutional layers with max-pooling for downsampling
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear layers for further processing
        self.fc1 = nn.Linear(16 * 7 * 7, 256)  # Adjust the input size based on your architecture
        self.relu4 = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Flatten the output before passing it through linear layers
        x = x.view(x.size(0), -1)
        
        x = self.relu4(self.fc1(x))
        
        return x


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class BYOLTrainer():
    def __init__(self, config):
        self.config = config
        self.gpu = 6
        
        
        self.logger = log(path=self.config['log_dir'], file=f"{self.config['exp_name']}.logs")
        self.logger.info('Training data Info:')
        self.logger.info(self.config)

        self.total_epochs = self.config['optimizer']['total_epochs']
        self.warmup_epochs = self.config['optimizer']['warmup_epochs']

        self.train_batch_size = self.config['data']['train_batch_size']
        self.val_batch_size = self.config['data']['val_batch_size']

        self.num_examples = self.config['data']['num_examples']
        self.warmup_steps = self.warmup_epochs * self.num_examples // self.train_batch_size    
        self.total_steps = self.total_epochs * self.num_examples // self.train_batch_size

        base_lr = self.config['optimizer']['base_lr']/self.train_batch_size
        self.max_lr = base_lr * self.train_batch_size 
        self.base_mm = self.config['model']['base_momentum']
        self.ckpt_path = self.config['checkpoint']['ckpt_path']

        self.resume_path = self.config['checkpoint']['resume_path']
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        """log tools in the running phase"""
        self.steps = 0
        self.log_step = self.config['log']['log_step']
        self.vae_ckpt_path = '/workspace/mayankk/Radiomics/BERT-BYOL/trainer/vq_vae_v2_mimic_100_final.pth'
        self.vae = vae
        
        self.construct_model()
        self.save_epoch = self.config['checkpoint']['save_epoch']

    def construct_model(self):
        
        self.data_ins = ImageNetLoader(self.config)
        dataset = self.config['data']['dataset']
        self.logger.info(f'dataset is {dataset}')
        if dataset=='mimic':
            self.train_loader = self.data_ins.mimic_loader(self.train_batch_size)
        elif dataset== 'NIH':
            self.train_loader = self.data_ins.NIH_loader(self.train_batch_size)
        elif dataset== 'PC':
            self.train_loader = self.data_ins.PC_loader(self.train_batch_size)
        else:
            self.train_loader = self.data_ins.get_loader(self.train_batch_size)

        self.logger.info("init byol model!")
        net = VAE_BYOLModel(self.config)
        self.model = net.to(self.device)
        self.logger.info(self.model)
        
        vae_checkpoint = torch.load(self.vae_ckpt_path,map_location=self.device)
        self.vae.load_state_dict(vae_checkpoint['model'])
        self.vae = self.vae.to(self.device)
        self.vae_projection = VAE_Projection().to(self.device)
        
        self.logger.info(self.vae)
        self.logger.info(self.vae_projection)
        

        self.logger.info("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        params = params_util.collect_params([self.model.online_network, self.model.predictor,
                                            self.vae_projection],
                                            exclude_bias_and_bn=exclude_bias_and_bn)
        self.optimizer = LARS(params, lr=self.max_lr,
                              momentum=momentum, weight_decay=weight_decay)
        
        self.mse_criterion = nn.MSELoss()


    def resume_model(self, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logger.info("--> No loaded checkpoint!")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch})")

    # save snapshots
    def save_checkpoint(self, epoch):
        if epoch % self.save_epoch == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'online': self.model.online_network.encoder.state_dict(),
                     'target': self.model.target_network.encoder.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     }
            torch.save(state, self.ckpt_path)

    def adjust_learning_rate(self, step):
        """learning rate warm up and decay"""
        max_lr = self.max_lr
        min_lr = 1e-3 * self.max_lr
        if step < self.warmup_steps:
            lr = (max_lr - min_lr) * step / self.warmup_steps + min_lr
        else:
            lr = min_lr + 0.5 * \
                (max_lr - min_lr) * (1 +
                                     np.cos((step - self.warmup_steps) * np.pi / self.total_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_mm(self, step):
        self.mm = 1 - (1 - self.base_mm) * \
            (np.cos(np.pi * step / self.total_steps) + 1) / 2
        
        
    

    def forward_loss(self, preds, targets):
        bz = preds.size(0)
        preds_norm = F.normalize(preds, dim=1)
        targets_norm = F.normalize(targets, dim=1)
        loss = 2 - 2 * (preds_norm * targets_norm).sum() / bz
        return loss

    def train_epoch(self, epoch):
        batch_time = eval_util.AverageMeter()
        data_time = eval_util.AverageMeter()
        forward_time = eval_util.AverageMeter()
        backward_time = eval_util.AverageMeter()
        log_time = eval_util.AverageMeter()
        loss_meter = eval_util.AverageMeter()

        self.model.train()

        end = time.time()
        self.data_ins.set_epoch(epoch)

        prefetcher = data_prefetcher(self.train_loader)
        images,vae_image = prefetcher.next()
        i = 0
        while images is not None:
            i += 1
            self.adjust_learning_rate(self.steps)
            self.adjust_mm(self.steps)
            self.steps += 1

            assert images.dim() == 5, f"Input must have 5 dims, got: {images.dim()}"
            view1 = images[:, 0, ...].contiguous()
            view2 = images[:, 1, ...].contiguous()

            # forward
            q1,q2,target_z1,target_z2,q_map1,q_map2 = self.model(view1, view2, self.mm)
            
            enc = self.vae._pre_vq_conv(self.vae._encoder(vae_image))
#             print('enc' ,enc.shape)
#             _, enc_quantize, _, _ = self.vae._vq_vae(enc)
            embd = self.vae_projection(enc)
    
            z_recon1 = self.vae._decoder(q_map1)
            z_recon2 = self.vae._decoder(q_map2)
    
            
            loss1 = self.forward_loss(q1, target_z2)
            loss2 = self.forward_loss(q2, target_z1)
            loss3 = self.forward_loss(q1, embd)
            loss4 = self.forward_loss(q2, embd)
            
            loss5 = self.forward_loss(target_z1, embd)
            loss6 = self.forward_loss(target_z2, embd)
            
            loss7 = self.mse_criterion(vae_image,z_recon1)
            loss8 = self.mse_criterion(vae_image,z_recon2)
            
            NC_loss = (loss1+loss2+loss3+loss4+loss5+loss6)/6
            MSE_loss = (loss7+loss8)/2
            
            loss = NC_loss + MSE_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), view1.size(0))

            # Print log info
            if self.steps % self.log_step == 0:
                self.logger.info(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                                f'Step {self.steps}\t'
                                f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}\t'
                                f'mm {round(self.mm, 5)}\t'
                                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t')
            images, _ = prefetcher.next()

