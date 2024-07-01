import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.l2(x)
        return x

class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # backbone
        pretrained = config['model']['backbone']['pretrained']
        self.pretrained = pretrained
        net_name = config['model']['backbone']['type']
        base_encoder = models.__dict__[net_name](pretrained=self.pretrained)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])
        input_dim = config['model']['projection']['input_dim']
        
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        output_size = (1,1)
        self.projetion = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self._initialize_weights(self.projetion)
    
    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0) 
        
    def forward(self, x):
        ftrs = self.encoder(x)
        pool = self.pool(ftrs)
        flat = torch.flatten(pool, 1)
        z    = self.projetion(flat)    
        return ftrs,z


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        # predictor
        input_dim = config['model']['predictor']['input_dim']
        hidden_dim = config['model']['predictor']['hidden_dim']
        output_dim = config['model']['predictor']['output_dim']
        self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self._initialize_weights(self.predictor)
        
    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)     

    def forward(self, x):
        return self.predictor(x)



