# -*- coding:utf-8 -*-
import time
import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from model import ModelVQ
from optimizer import LARS
from data import NihDataLoader
from utils import params_util, logging_util, eval_util
from utils.data_prefetcher import data_prefetcher
from utils.logging_util import log
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# +
class VQTrainer():
    def __init__(self, config):
        self.config = config
        self.gpu = self.config['gpu']
        
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
        
        self.save_epoch = self.config['checkpoint']['save_epoch']
        self.exp_name = self.config['exp_name']+'_'+self.config['model']['backbone']['type']+'_'+str(self.config['data']['train_batch_size'])
        self.save_path = os.path.join(self.ckpt_path,self.exp_name)
        os.makedirs(self.save_path, exist_ok=True)

        """log tools in the running phase"""
        self.logger = log(path=self.save_path, file=f"{self.exp_name}.logs")
        self.logger.info('Training data Info:')
        self.logger.info(self.config)
        self.steps = 0
        self.log_step = self.config['log']['log_step']
        self.base_mu = 0.1
        self.construct_model()

    def construct_model(self):
        
        self.data_ins = NihDataLoader(self.config)
        dataset = self.config['data']['dataset']
        self.logger.info(f'dataset is {dataset}')
        self.train_loader = self.data_ins.NIH_loader(self.train_batch_size)
        
        self.logger.info("init byol model!")
        net = ModelVQ(self.config)
        self.model = net.to(self.device)
        self.logger.info(self.model)

        self.logger.info("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        params = params_util.collect_params([self.model.online_network, self.model.predictor,
                                            self.model.codebook,self.model.fc_fuse,
                                            self.model.sa],
                                            exclude_bias_and_bn=exclude_bias_and_bn)
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
            model_state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'online': self.model.online_network.encoder.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     }
            online_state = {'online': self.model.online_network.encoder.state_dict()}
            
            SAVE_PATH1 = os.path.join(self.save_path, f'{self.exp_name}_model.pth')
            SAVE_PATH2 = os.path.join(self.save_path, f'{self.exp_name}_{epoch}.pth')
            torch.save(model_state, SAVE_PATH1)
            torch.save(online_state, SAVE_PATH2)

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
        
    def adjust_mu(self, step):
        self.mu = 1 - (1 - self.base_mu) * \
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
        ce_loss_meter = eval_util.AverageMeter()
        eq_loss_meter = eval_util.AverageMeter()
        perp_meter = eval_util.AverageMeter()
        
        self.model.train()
        end = time.time()
        self.data_ins.set_epoch(epoch)

        prefetcher = data_prefetcher(self.train_loader)
        images,_ = prefetcher.next()
        i = 0
        while images is not None:
            i += 1
            self.adjust_learning_rate(self.steps)
            self.adjust_mm(self.steps)
            self.adjust_mu(self.steps)
            
            self.steps += 1

            assert images.dim() == 5, f"Input must have 5 dims, got: {images.dim()}"
            view1 = images[:, 0, ...].contiguous()
            view2 = images[:, 1, ...].contiguous()

            # forward
            q1,q2,target_z1,target_z2,fuse1,fuse2,e_q_loss,perplexity = self.model(view1, view2, self.mm)
            
            
            loss1 = self.forward_loss(q1, target_z2)
            loss2 = self.forward_loss(q2, target_z1)
            vq_loss1 = self.forward_loss(q2, fuse1)
            vq_loss2 = self.forward_loss(q1, fuse2)                        
            ce_loss = (loss1 + loss2+vq_loss1+vq_loss2)/4.0
            loss = (10*ce_loss +eq_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), view1.size(0))
            ce_loss_meter.update(ce_loss.item(), view1.size(0))
            eq_loss_meter.update(eq_loss.item(), view1.size(0))
            perp_meter.update(perplexity,view1.size(0))

            # Print log info
            if self.steps % self.log_step == 0:
                self.logger.info(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]&'
                                f'Step {self.steps}--'
                                f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}--'
                                f'mm {round(self.mm, 5)}--'
                                f'Loss {loss_meter.val:.4f}--'
                                f'CE_Loss {ce_loss_meter.val:.4f}--'
                                f'EQ_Loss {eq_loss_meter.val:.4f}-'
                                f'perp {perp_meter.val:.4f}'
                                )
            images, _ = prefetcher.next()

