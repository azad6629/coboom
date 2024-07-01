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

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear transformations for queries, keys, and values for each head
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

        # Output linear transformation
        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x1, x2):
        batch_size, _, height, width = x1.size()
        x1 = x1.view(batch_size, -1, x1.size(1))
        batch_size, _, height, width = x2.size()
        x2 = x2.view(batch_size, -1, x2.size(1))
        seq_len = x1.size(1)

        queries = self.query_linear(x1).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key_linear(x2).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_linear(x2).view(batch_size, seq_len, self.num_heads, self.head_dim)

        scores = torch.einsum("bqhd,bkhd->bhqk", queries, keys) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.einsum("bhqk,bkhd->bqhd", attention_weights, values)
        concatenated_attentions = attended_values.reshape(batch_size, seq_len, -1)
        output = self.output_linear(concatenated_attentions)
        output = output.view(batch_size, -1, height, width)
        return output


class ModelVQ(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.online_network = EncoderwithProjection(config)
        self.target_network = EncoderwithProjection(config)
        self.predictor      = Predictor(config)
        num_embeddings      = config['cb']['num_embeddings']
        embd_dims           = config['cb']['embd_dims']
        comit_cost          = config['cb']['comit_cost']

        self.codebook = VectorQuantizer(num_embeddings, embd_dims, comit_cost)
        self.sa       = MultiHeadCrossAttention(embd_dims, num_heads=8)
        
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
            
            feat1 ,target_z1 = self.target_network(view1)               
            target_z1 = target_z1.detach().clone()
            feat2,target_z2 = self.target_network(view2)    
            target_z2 = target_z2.detach().clone()
            
        quantized1, e_q_loss1,perplexity1,_ = self.codebook(feat1)      
        quantized1 = self.sa(quantized1,feat1)
        quant1 = F.adaptive_avg_pool2d(quantized1, (1, 1)).reshape(bs, -1)
        fuse1 = self.fc_fuse(quant1)
        
        quantized2, e_q_loss2,perplexity2,_ = self.codebook(feat2)
        quantized2 = self.sa(quantized2,feat2)
        quant2 = F.adaptive_avg_pool2d(quantized2, (1, 1)).reshape(bs, -1)
        fuse2 = self.fc_fuse(quant2)
        
        perplexity = (perplexity1+perplexity2)/2.0  
        eqloss     = (e_q_loss1+e_q_loss2)/2.0  

        return q1,q2,target_z1,target_z2,fuse1,fuse2,eqloss,perplexity
