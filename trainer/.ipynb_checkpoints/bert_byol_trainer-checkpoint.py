# -*- coding:utf-8 -*-
import time
import datetime
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from model import BYOLModel
from optimizer import LARS
from data import ImageNetLoader
from utils import params_util, logging_util, eval_util
from utils.data_prefetcher_bert import data_prefetcher
from utils.logging_util import log
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# +
class TextEncoder(nn.Module):
    def __init__(self, config ):
        super().__init__()
        
        model_name=config['text_encoder_model']
        pretrained=config['text_pretrained']
        trainable=config['text_trainable']
        
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        if trainable:
            for p in self.model.parameters():
                p.requires_grad = trainable
                
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask,return_dict=False)
        return pooled_output
    
class ProjectionHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        embedding_dim= config['text_embedding']
        projection_dim=config['projection_dim']
        dropout=config['dropout']
        
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu       = nn.GELU()
        self.fc         = nn.Linear(projection_dim, projection_dim)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


# -

class BYOLTrainer():
    def __init__(self, config):
        self.config = config
        self.gpu = 5
        
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
            
        self.construct_model()
        self.save_epoch = self.config['checkpoint']['save_epoch']

        """log tools in the running phase"""
        self.steps = 0
        self.log_step = self.config['log']['log_step']

    def construct_model(self):
        
        self.data_ins = ImageNetLoader(self.config)
        dataset = self.config['data']['dataset']
        self.logger.info(f'dataset is {dataset}')
        if dataset=='mimic':
            self.train_loader = self.data_ins.mimic_loader(self.train_batch_size)
        else:
            self.train_loader = self.data_ins.get_loader(self.train_batch_size)

        self.logger.info("init byol model!")
        net = BYOLModel(self.config)
        self.model = net.to(self.device)
        self.logger.info(self.model)
        
        self.text_encoder  = TextEncoder(self.config).to(self.device)
        self.text_projection  = ProjectionHead(self.config).to(self.device)
        self.logger.info(self.text_encoder)
        self.logger.info(self.text_projection)


        self.logger.info("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        params = params_util.collect_params([self.model.online_network, self.model.predictor,self.text_encoder,
                                             self.text_projection],exclude_bias_and_bn=exclude_bias_and_bn)
        self.optimizer = LARS(params, lr=self.max_lr,momentum=momentum, weight_decay=weight_decay)


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

    def train_epoch(self, epoch, printer=print):
        batch_time = eval_util.AverageMeter()
        data_time = eval_util.AverageMeter()
        forward_time = eval_util.AverageMeter()
        backward_time = eval_util.AverageMeter()
        log_time = eval_util.AverageMeter()
        loss_meter = eval_util.AverageMeter()

        self.model.train()
        self.text_projection.train()

        end = time.time()
        self.data_ins.set_epoch(epoch)

        prefetcher = data_prefetcher(self.train_loader)
        images,input_ids,attention_mask = prefetcher.next()
      
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
            q1,q2, target_z1,target_z2 = self.model(view1, view2, self.mm)
            
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings   = self.text_projection(text_features)
            
            
            loss1 = self.forward_loss(q1, target_z2)
            loss2 = self.forward_loss(q2, target_z1)
            loss3 = self.forward_loss(q1, text_embeddings)
            loss4 = self.forward_loss(q2, text_embeddings)
            
            loss5 = self.forward_loss(target_z1, text_embeddings)
            loss6 = self.forward_loss(target_z2, text_embeddings)
            loss = (loss1+loss2+loss3+loss4+loss5+loss6)/6

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
            images,input_ids,attention_mask = prefetcher.next()

