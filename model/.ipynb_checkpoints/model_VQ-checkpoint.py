#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules_vq import EncoderwithProjection, Predictor

class VectorQuantizer(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, commitment_cost,decay=0.99,epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self._ema_w.data.normal_()
        
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs):        
        bs,channel = inputs.shape[0],inputs.shape[1]
        # convert inputs from BCHW -> BHWC to do qunatization in channel space
        inputs = inputs.permute(0, 2, 3, 1).contiguous() 
        input_shape = inputs.shape
        # Flatten input
        flat_x = inputs.view(-1, self.embedding_dim)
        flat_x = F.normalize(flat_x, p=2, dim=1)
        weight = self.embedding.weight
        weight = F.normalize(weight, p=2, dim=1)
        
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True)  
                          + torch.sum(weight ** 2, dim=1) 
                          - 2. * torch.matmul(flat_x, weight.t()))  # [N, M]
        
        encoding_indices = torch.argmin(distances, dim=1) 
        """Returns embedding tensor for a batch of indices."""
        encoding_indices = encoding_indices.unsqueeze(1) 
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encoding_indices, 0)
            
        # Laplace smoothing of the cluster size
        n = torch.sum(self.ema_cluster_size.data)
        self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon)
                                 / (n + self.num_embeddings * self.epsilon) * n)
            
        dw = torch.matmul(encodings.t().to(flat_x.dtype), flat_x)
        
        self._ema_w = nn.Parameter(self._ema_w * self.decay + (1 - self.decay) * dw)
        self.embedding.weight = nn.Parameter(self._ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        encoding_indices = encoding_indices#.view(bs,channel)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        

        return quantized,loss,perplexity,encodings


class Info_Inter_Module_1D(nn.Module):
    def __init__(self, M=2, k_size=3):
        """
        The local cross-channel information interaction attention mechanism
        for the fusion of multi-dimensional features.

        :param channel: the channels of the input feature map
        :param M: the number of input features
        :param k_size: the kernel size of the 1D conv, determining the scale of information interaction
        """
        super().__init__()
        self.M = M

        # Remove AdaptiveAvgPool2d for 1D features
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.convs = nn.ModuleList([])
        for i in range(self.M):
            # Replace Conv2d with Conv1d
            self.convs.append(
                nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size, channel = x1.shape
        # Concatenate along the channel dimension
        feats = torch.cat([x1, x2], dim=1)
        # Reshape to [batch_size, M, channel]
        feats = feats.view(batch_size, self.M, channel)

        # Sum along the M dimension
        feats_S = torch.sum(feats, dim=1)
        feats_G = feats_S.unsqueeze(1)
        # Convolutional operations
        attention_vectors = [conv(feats_G).transpose(1, 2) for conv in self.convs]
        attention_vectors = torch.cat(attention_vectors, dim=1)        
        attention_vectors = attention_vectors.view(batch_size, self.M, channel)        
        attention_vectors = self.softmax(attention_vectors)
        # Element-wise multiplication and sum along the M dimension
        feats_o = torch.sum(feats * attention_vectors, dim=1)
        return feats_o


# +
class BYOLModelVQL(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.online_network = EncoderwithProjection(config)
        self.target_network = EncoderwithProjection(config)
        self.predictor      = Predictor(config)
        if config['model']['backbone']['type'] == 'resnet50':
            num_embeddings  = config['cb']['num_embeddings_r50']
        if config['model']['backbone']['type'] == 'resnet18':
            num_embeddings  = config['cb']['num_embeddings_r18']
        embd_dims           = config['cb']['embd_dims']
        comit_cost          = config['cb']['comit_cost']

        self.codebook = VectorQuantizer(num_embeddings, embd_dims, comit_cost)

#         self.fc_fuse = nn.Sequential(nn.Linear(2048 * 2, 2048),
#                                      nn.ReLU(True),
#                                      nn.Linear(2048, 256)
#                                     )
    
#         self.IIM = Info_Inter_Module_1D()
        self.fc_fuse = nn.Sequential(nn.Linear(embd_dims, 1024),
                                     nn.ReLU(True),
                                     nn.Linear(1024, 256)
                                    )

        self._initializes_target_network()
        self.use_momentum = config['model']['use_momentum']

    @torch.no_grad()
    def _initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient
  
    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.mul_(mm).add_(param_q.data,alpha=1. - mm )
    
    def adaptivefusion(self,q,f,mu):
        fused = torch.stack([mu * q, (1-mu) * f], dim=2)
        return fused

    def forward(self, view1, view2, mm):
        bs = view1.shape[0]
        _,q1 = self.online_network(view1)
        q1 = self.predictor(q1)
        _,q2 = self.online_network(view2)
        q2 = self.predictor(q2)
                
        # target network forward
        with torch.no_grad():
            if self.use_momentum:
                self._update_target_network(mm)
            else:
                self.target_network=self.online_network
                
            feat1,target_z1 = self.target_network(view1)               
            target_z1 = target_z1.detach().clone()
            feat1 = feat1.detach().clone()
            
            quantized1, e_q_loss1,perplexity1,_ = self.codebook(feat1)      
            quantized1 = F.adaptive_avg_pool2d(quantized1, (1, 1)).reshape(bs, -1)
#             feat1 = F.adaptive_avg_pool2d(feat1, (1, 1)).reshape(bs, -1)
#             fuse1 = self.IIM(quantized1,feat1)
#             fuse1 = self.adaptivefusion(quantized1,feat1,mu).reshape(bs, -1)
#             fuse1 = torch.stack([quantized1, feat1], dim=2).reshape(bs, -1)  # b,d,2
            fuse1 = self.fc_fuse(quantized1)
            
            feat2,target_z2 = self.target_network(view2)    
            target_z2 = target_z2.detach().clone()
            feat2 = feat2.detach().clone()
            quantized2, e_q_loss2,perplexity2,_ = self.codebook(feat2)
            quantized2 = F.adaptive_avg_pool2d(quantized2, (1, 1)).reshape(bs, -1)
#             feat2 = F.adaptive_avg_pool2d(feat2, (1, 1)).reshape(bs, -1)

#             fuse2 = self.IIM(quantized2,feat2)
#             fuse2 = self.adaptivefusion(quantized2,feat2,mu).reshape(bs, -1) 
#             fuse2 = torch.stack([quantized2, feat2], dim=2).reshape(bs, -1)  # b,d,2
            fuse2 = self.fc_fuse(quantized2)
            perplexity = (perplexity1+perplexity2)/2.0
        return q1,q2,target_z1,target_z2,fuse1,fuse2,e_q_loss1,e_q_loss2,perplexity

# +
# import yaml

# with open('/workspace/data/VQSSL/config/train_config_mimic.yaml', 'r') as f:
#     config = yaml.safe_load(f)


# model = BYOLModelVQL(config)
# print(model)


# x = torch.rand(4,3,224,224)

# q1,q2,target_z1,target_z2,fuse1,fuse2,e_q_loss1,e_q_loss2,perplexity = model(x,x,0.996)
# -

