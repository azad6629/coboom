#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class EncoderwithProjection(nn.Module):
    def __init__(self,config,pretrained):
        super().__init__()
        # backbone
        self.pretrained = pretrained
        net_name = config['model']['backbone']['type']
        
        base_encoder = models.__dict__[net_name](pretrained=self.pretrained)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])
        
            
    def forward(self, x):
        x = self.encoder(x)
        return x

# +
# x = torch.rand(1,3,224,224).cuda()
# model = EncoderwithProjection().cuda()
# out = model(x)

# +
# out.shape

# +
# class Predictor(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         # predictor
#         input_dim = config['model']['predictor']['input_dim']
#         hidden_dim = config['model']['predictor']['hidden_dim']
#         output_dim = config['model']['predictor']['output_dim']
#         self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

#     def forward(self, x):
#         return self.predictor(x)
# -

class Codebook(nn.Module):
    def __init__(self, config):
                 
        super(Codebook, self).__init__()
        
        self._embedding_dim = config['cb']['embedding_dim']
        self._num_embeddings = config['cb']['num_embeddings']
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = config['cb']['commitment_cost']
        
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(self._num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = config['cb']['decay']
        self._epsilon = float(config['cb']['epsilon'])

    def forward(self, inputs):
        x = inputs
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
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
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            
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
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings



