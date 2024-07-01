#-*- coding:utf-8 -*-
import torch
from .basic_modules_VAE import EncoderwithProjection, Predictor

class VAE_BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.online_network = EncoderwithProjection(config,pretrained=False)
        self.target_network = EncoderwithProjection(config,pretrained=False)
        self.predictor = Predictor(config)

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
            

    def forward(self, view1, view2, mm): 
        q1,q_map1 = self.online_network(view1)    #self.predictor(
        q1        = self.predictor(q1)
        q2,q_map2 = self.online_network(view2)
        q2        = self.predictor(q2)

        # target network forward
        with torch.no_grad():
            if self.use_momentum:
                self._update_target_network(mm)
            else:
                self.target_network=self.online_network
                
            target_z1,_ = self.target_network(view1)    
            target_z1 = target_z1.detach().clone()
            
            target_z2,_ = self.target_network(view2)    
            target_z2 = target_z2.detach().clone()
            

        return q1,q2,target_z1,target_z2,q_map1,q_map2
