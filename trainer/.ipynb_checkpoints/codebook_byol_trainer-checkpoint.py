# -*- coding:utf-8 -*-
import time
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from .v1_vq_vae import model as vae
# from .v1_vq_vae_nih import model as vae

from model import BYOLModel_CB
from optimizer import LARS
from data import ImageNetLoader
from utils import params_util, logging_util, eval_util
from utils.data_prefetcher import data_prefetcher
# from utils.data_prefetcher_bert import data_prefetcher
from utils.logging_util import log
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# +
# import torch
# import torch.nn as nn

# class ReshapeModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ReshapeModule, self).__init__()
#         self.proj =nn.Sequential(nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
#                    nn.BatchNorm2d(512),
#                      nn.ReLU(),
#                      nn.MaxPool2d(kernel_size=2, stride=2),

#                      nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#                      nn.BatchNorm2d(1024),
                                 
#                      nn.ReLU(),
#                      nn.MaxPool2d(kernel_size=2, stride=2),
                    
#                      nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
#                      nn.BatchNorm2d(2048),

#                      nn.ReLU(),
#                      nn.MaxPool2d(kernel_size=2, stride=2)
#                                 )

#     def forward(self, x):
#         x = self.proj(x)
#         return x

# # Assuming you have a feature matrix x of shape [1, 128, 56, 56]
# x = torch.rand((1, 128, 56, 56))

# # Reshape the feature matrix
# reshape_module = ReshapeModule(in_channels=128, out_channels=2048)
# reshaped_x = reshape_module(x)

# # Print the shapes for verification
# print("Original shape:", x.shape)
# print("Reshaped shape:", reshaped_x.shape)


# +
# class VAE_Projection(nn.Module):
#     def __init__(self):
#         super(VAE_Projection, self).__init__()
#         # Convolutional layers with max-pooling for downsampling
    
#         self.proj =nn.Sequential(nn.Conv2d(128, 512, kernel_size=3, padding=1),
#                    nn.BatchNorm2d(512),
#                    nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=2, stride=2),

#                      nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#                      nn.BatchNorm2d(1024),
                                 
#                      nn.ReLU(),
#                      nn.MaxPool2d(kernel_size=2, stride=2),
                    
#                      nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
#                      nn.BatchNorm2d(2048),

#                      nn.ReLU(),
#                      nn.MaxPool2d(kernel_size=2, stride=2)
#                                 )

#     def forward(self, x):
#         x = self.proj(x)
#         x = x.view(x.size(0),x.size(1),-1)

#         return x


class VAE_Projection(nn.Module):
    def __init__(self):
        super(VAE_Projection, self).__init__()
        # Convolutional layers with max-pooling for downsampling
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
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


# +
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

        """log tools in the running phase"""
        self.steps = 0
        self.log_step = self.config['log']['log_step']
        self.vae_ckpt_path = '/workspace/mayankk/Radiomics/BERT-BYOL/trainer/vq_vae_v1_mimic_100_final.pth'
#         self.vae_ckpt_path = '/workspace/mayankk/Radiomics/BERT-BYOL/trainer/vq_vae_v1_NIH_100_final.pth'
        self.vae = vae
        print('VAE encoder laoded success')
        
        self.construct_model()
        self.save_epoch = self.config['checkpoint']['save_epoch']
        self.exp_name = self.config['checkpoint']['exp_name']
        self.save_path = os.path.join(self.ckpt_path,self.exp_name)
        os.makedirs(self.save_path, exist_ok=True)

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
        net = BYOLModel_CB(self.config)
        self.model = net.to(self.device)
        self.logger.info(self.model)
        
        vae_checkpoint = torch.load(self.vae_ckpt_path)
        self.vae.load_state_dict(vae_checkpoint['model'])
        self.vae = self.vae.to(self.device)
        self.vae_projection = VAE_Projection().to(self.device)
        
        self.logger.info(self.vae)
        self.logger.info(self.vae_projection)
        

        self.logger.info("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        params = params_util.collect_params([self.model.online_network, 
#                                              self.model.online_codebook,
#                                              self.model.target_codebook,
                                             self.vae_projection],
                                            exclude_bias_and_bn=exclude_bias_and_bn)
        
#         self.optimizer=torch.optim.Adam(params, lr=self.max_lr,betas=(0.9, 0.999),eps = 1e-08,
#                                         weight_decay = weight_decay) 

        self.optimizer = LARS(params, lr=self.max_lr,
                              momentum=momentum, weight_decay=weight_decay)


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
#                      'target': self.model.target_network.encoder.state_dict(),
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

    def forward_loss_global(self, preds, targets):
        bz = preds.size(0)
        preds_norm = F.normalize(preds, dim=1)
        targets_norm = F.normalize(targets, dim=1)
        loss = 2 - 2 * (preds_norm * targets_norm).sum() / bz
        return loss
    
    def calculate_entropy_loss(self,vectors):
            
            bs, d, m = vectors.size()
            # Reshape to (bs * d, m)
            vectors_flat = vectors.view(bs * d, -1)
            # Compute entropy of the distribution
            distribution_entropy = -torch.sum(vectors_flat * torch.log(vectors_flat + 1e-12), dim=1)
            # Penalize low entropy
            diversity_penalty = torch.mean(F.relu(0.5 - distribution_entropy))  # Adjust the threshold as needed
            return diversity_penalty

    
    def entropy_based_diversity_loss_individual(self,x, y):
        
        # Calculate entropy-based diversity loss individually for x and y
        diversity_penalty_x = self.calculate_entropy_loss(x)
        diversity_penalty_y = self.calculate_entropy_loss(y)

        return diversity_penalty_x, diversity_penalty_y

    
    
    def cost_matrix_batch_torch(self, x, y):
        "Returns the cosine distance batchwise"
        # x is the image feature: bs * d * m 
        # y is the image feature: bs * d * n
        # return: bs * n * m

        bs = list(x.size())[0]
        D = x.size(1)
        assert(x.size(1)==y.size(1))
        x = x.contiguous().view(bs, D, -1) # bs * d * m^2
        y = y.contiguous().view(bs, D, -1) 
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)

        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)#.transpose(1,2)
        # print(cos_dis.shape)
        cos_dis = 1 - cos_dis # to minimize this value
        return cos_dis.transpose(2,1)
    
    def forward_loss_local(self, preds, targets):
        dist = self.cost_matrix_batch_torch(preds,targets)
        beta = 0.1
        min_score = dist.min()
        max_score = dist.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(dist - threshold)
        cos_loss = -torch.mean(cos_dist)  
        diversity_penalty_x, diversity_penalty_y = self.entropy_based_diversity_loss_individual(preds, targets)

#         loss = cos_loss + 0.5*diversity_penalty_x
        return cos_loss
    def feature_based_matching_loss(self,x, y, gamma=0.5):
        """
        Feature-based matching loss function.

        Args:
        - x: Tensor of shape [bs, 2048, 49] representing the feature maps.
        - y: Tensor of shape [bs, 2048, 49] representing the target feature maps.
        - gamma: Top-γ parameter to control the number of pairs to consider.

        Returns:
        - loss: Scalar tensor representing the feature-based matching loss.
        """

        bs, channels, num_features = x.size()

        # Flatten the spatial dimensions
        x_flat = x.view(bs, channels, -1)
        y_flat = y.view(bs, channels, -1)

        # Reshape to [bs * num_features, channels]
        x_flat = x_flat.permute(0, 2, 1).contiguous().view(-1, channels)
        y_flat = y_flat.view(-1, channels)

        # Compute pairwise L2 distances
        distances = torch.cdist(x_flat, y_flat, p=2)

        # Find the nearest neighbors
        _, indices = torch.topk(distances, k=1, dim=1, largest=False)

        # Calculate the loss for each pair
        loss_per_pair = distances[torch.arange(indices.size(0)), indices.view(-1)]

        # Take the top-γ pairs
        _, top_indices = torch.topk(loss_per_pair, k=int(gamma * num_features), largest=False)

        # Calculate the mean loss for the selected pairs
        loss = torch.mean(loss_per_pair[top_indices])

        return loss


    def train_epoch(self, epoch):
        batch_time = eval_util.AverageMeter()
        data_time = eval_util.AverageMeter()
        forward_time = eval_util.AverageMeter()
        backward_time = eval_util.AverageMeter()
        log_time = eval_util.AverageMeter()
        loss_meter = eval_util.AverageMeter()
        
        local_loss_meter = eval_util.AverageMeter()
        global_loss_meter = eval_util.AverageMeter()
#         cbl1_meter = eval_util.AverageMeter()
#         cbl2_meter = eval_util.AverageMeter()

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
            q1, q1_ftr, q2, q2_ftr, z1_ftr, target_z1, z2_ftr, target_z2  = self.model(view1, view2, self.mm)
            
            
                        
            enc = self.vae._pre_vq_conv(self.vae._encoder(vae_image))
            _, enc_quantize, _, _ = self.vae._vq_vae(enc)
            embd = self.vae_projection(enc_quantize)
            
#             print(q1.shape,q1_ftr.shape,embd.shape)
#             print(target_z1.shape,z1_ftr.shape,embd.shape)
                       
#             local_loss1 = self.forward_loss_local(q1_ftr, z2_ftr)
#             local_loss2 = self.forward_loss_local(q2_ftr, z1_ftr)
            local_loss1 = self.feature_based_matching_loss(q1_ftr,z2_ftr)
            local_loss2 = self.feature_based_matching_loss(q2_ftr,z1_ftr)

            
#             local_loss3 = self.forward_loss_local(q1_ftr, embd)
#             local_loss4 = self.forward_loss_local(q2_ftr, embd)
            
            local_loss = (local_loss1+local_loss2)/2.0 #+local_loss3+local_loss4)/2.0
            
            global_loss1 = self.forward_loss_global(q1, target_z2)
            global_loss2 = self.forward_loss_global(q2, target_z1)
            global_loss3 = self.forward_loss_global(q1, embd)
            global_loss4 = self.forward_loss_global(q2, embd)
            global_loss = (global_loss1+global_loss2+global_loss3+global_loss4)/4.0
            
#             loss = 0.7*(global_loss) + 0.3*(local_loss)
            loss = global_loss + local_loss
#             loss = 0.3*(global_loss) + 0.7*(local_loss)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), view1.size(0))
            
            local_loss_meter.update(local_loss.item(), view1.size(0))
            global_loss_meter.update(global_loss.item(), view1.size(0))
#             cbl1_meter.update(cbl1.item(), view1.size(0))
#             cbl2_meter.update(cbl2.item(), view1.size(0))

            # Print log info
            if self.steps % self.log_step == 0:
                self.logger.info(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]--'
                                f'Step {self.steps}--'
                                f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}--'
                                f'mm {round(self.mm, 5)}--'
                                f'Loss {loss_meter.val:.4f}--'
                                f'L_Loss {local_loss_meter.val:.2f}--'
                                f'G_Loss {global_loss_meter.val:.2f}' 
                                 
                                )
                        
            images, _ = prefetcher.next()

