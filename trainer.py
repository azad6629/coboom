import os
import time
import torch
import random
import numpy as np
from optimizer import LARS
from data import DataLoader
from model import CoBoom
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import params_util, eval_util
from utils.logging_util import log

class Trainer():
    def __init__(self, config):
        self.config = config
        self.gpu = self.config['gpu']
        self.seed = self.config['seed']
        
        self.batch_size = self.config['data']['pre_bs']
        self.total_epochs = self.config['optimizer']['total_epochs']
        self.warmup_epochs = self.config['optimizer']['warmup_epochs']
        self.base_mm = self.config['model']['base_momentum']
        self._setup_logging()
        self.device = self._setup_device()
        self._setup_data_loaders()
        self._setup_learning_params()
        self.construct_model()        
        
    def _setup_logging(self):
        self.dataset = self.config['data']['dataset']
        save_path = os.path.join('./ckpt', self.config['model_name'].lower()+'_'+self.config['ver'])
        self.method_name = f"{self.config['model']['backbone']['type']}_{self.dataset}_{self.batch_size}_{self.total_epochs}"
        
        ckpt_path = os.path.join(save_path, self.method_name)
        self.config['checkpoint']['ckpt_path'] = ckpt_path
        os.makedirs(ckpt_path, exist_ok=True)
        
        self.logger = log(path=ckpt_path, file=f"{self.method_name}.logs")
        
        self.steps = 0
        self.total_training_time = 0
        self.log_step = self.config['checkpoint']['log_step']
        self.save_epoch = self.config['checkpoint']['save_epoch']    
    
    
    def _setup_learning_params(self):
        base_lr = float(self.config['optimizer']['base_lr']) / 256
        self.max_lr = base_lr * self.batch_size
    
    def _setup_data_loaders(self):
        data_ins = DataLoader(self.config)
        if self.dataset == 'NIH14':
            self.train_loader, self.valid_loader, _ = data_ins.GetNihDataset()
        elif self.dataset == 'Chex14':
            self.train_loader, self.valid_loader, self.test_loader = data_ins.GetChex14Dataset()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
            
        num_examples = self.config['data']['num_examples']
        self.warmup_steps = self.warmup_epochs * num_examples // self.batch_size    
        self.total_steps = self.total_epochs * num_examples // self.batch_size

    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(device)
            cudnn.benchmark = True
            return device
        return torch.device('cpu')
    
    def seed_everything(self,seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def construct_model(self):
        self.logger.info('Training data Info:')
        self.logger.info(self.config)
        self.logger.info("init model!")
        self.logger.info(f'dataset is {self.dataset}')
        self.model = CoBoom((self.config)).to(self.device)
        self.logger.info(self.model)
        self.logger.info("get optimizer!")
            
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        
        params = params_util.collect_params([self.model.online_network, 
                                 self.model.predictor,
                                 self.model.codebook, 
                                 self.model.df, 
                                 self.model.fc_fuse,                                 
                                 self.model.online_decoder],
                                exclude_bias_and_bn=exclude_bias_and_bn)
        
        self.optimizer = LARS(params, lr=self.max_lr,
                              momentum=momentum, weight_decay=weight_decay) 
        self.seed_everything(self.seed)

    def resume_model(self, resume=True):
        """Resume training from checkpoint if available"""
        self.start_epoch = 0
        if resume:
            checkpoint_path = os.path.join(self.config['checkpoint']['ckpt_path'], f"{self.method_name}.pth")
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.start_epoch = checkpoint['epoch']
                self.steps = checkpoint['steps']
                self.model.load_state_dict(checkpoint['model'], strict=True)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"--> Loaded checkpoint (epoch {self.start_epoch})")
                return
                
        self.logger.info("--> No checkpoint loaded, starting from scratch") 

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if epoch % self.save_epoch == 0:
            ckpt_path = self.config['checkpoint']['ckpt_path']
            
            # Full model state
            model_state = {
                'config': self.config,
                'epoch': epoch,
                'steps': self.steps,
                'model': self.model.state_dict(),
                'online': self.model.online_network.encoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            online_state = {
                            'epoch': epoch,
                            'online': self.model.online_network.encoder.state_dict()
                            }

            main_path = os.path.join(ckpt_path, f'{self.method_name}.pth')
            epoch_path = os.path.join(ckpt_path, f'{self.method_name}_{epoch}.pth')          
            torch.save(model_state, main_path)
            torch.save(online_state, epoch_path)
            self.logger.info(f"Checkpoint saved at epoch {epoch}")
            
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
    
    def recursive_to_device(self, inp, device):
        """Recursively send tensors to device"""
        if isinstance(inp, list):
            return [self.recursive_to_device(item, device) for item in inp]
        elif isinstance(inp, tuple):
            return tuple(self.recursive_to_device(item, device) for item in inp)
        elif isinstance(inp, dict):
            return {k: self.recursive_to_device(v, device) for k, v in inp.items()}
        elif isinstance(inp, torch.Tensor):
            return inp.to(device)
        else:
            return inp
        
    def train_epoch(self, epoch):
        loss_meter    = eval_util.AverageMeter()
        ce_loss_meter = eval_util.AverageMeter()
        eq_loss_meter = eval_util.AverageMeter()
        perp_meter    = eval_util.AverageMeter()
        recon_meter   = eval_util.AverageMeter()
        
        self.model.train()
        epoch_start_time = time.time()

        for i, (img, lbl) in enumerate(self.train_loader):
            self.adjust_learning_rate(self.steps)
            self.adjust_mm(self.steps)
            self.steps += 1
            img = self.recursive_to_device(img, self.device)
            _,_,orig_img = img
            q1,q2,x_recon1,x_recon2,target_z1,target_z2,fuse1,fuse2,eq_loss,perplexity = self.model(img, self.mm)
            
            loss1 = self.forward_loss(q1, target_z2)
            loss2 = self.forward_loss(q2, target_z1)
            vq_loss1 = self.forward_loss(q2, fuse1)
            vq_loss2 = self.forward_loss(q1, fuse2)
            
            ce_loss = (loss1 + loss2+vq_loss1+vq_loss2)/4.0
            
            recon_loss1 = F.mse_loss(x_recon1, orig_img) 
            recon_loss2 = F.mse_loss(x_recon2, orig_img)
            recon_error = (recon_loss1+recon_loss2)/2.0
                        
            loss = (ce_loss + eq_loss+ recon_error).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), img[0].size(0))
            ce_loss_meter.update(ce_loss.item(), img[0].size(0))
            eq_loss_meter.update(eq_loss.item(), img[0].size(0))
            recon_meter.update(recon_error.item(), img[0].size(0))
            perp_meter.update(perplexity,img[0].size(0))

            # Print log info
            if self.steps % self.log_step == 0:
                self.logger.info(f'Epoch: [{epoch}][{i+1}/{len(self.train_loader)}]&'
                                f'Step {self.steps}-'
                                f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}-'
                                f'mm {round(self.mm, 5)}-'
                                f'Loss {loss_meter.val:.2f}-'
                                f'CE_Loss {ce_loss_meter.val:.2f}-'
                                f'EQ_Loss {eq_loss_meter.val:.2f}-'
                                f'R_Loss {recon_meter.val:.2f}-'
                                f'perp {perp_meter.val:.2f}'
                                )
                
        epoch_training_time = (time.time() - epoch_start_time) / 60
        self.total_training_time += epoch_training_time
        self.logger.info(f"Epoch {epoch} completed in {epoch_training_time:.2f} minutes")
        
        # Log total training time at end of training
        if epoch == self.total_epochs:
            total_hours = self.total_training_time / 60
            self.logger.info(f"Total training time: {total_hours:.2f} hours")
        
        return {
            'loss': loss_meter.avg,
            'cl_loss': ce_loss_meter.avg,
            'q_loss': eq_loss_meter.avg,
            'r_loss': recon_meter.avg,
            'perplexity': perp_meter.avg
        } 
