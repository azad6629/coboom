#-*- coding:utf-8 -*-
import torch
from .basic_modules_CB2 import EncoderwithProjection, Codebook

# +
class BYOLModel_CB2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.online_network = EncoderwithProjection(config,pretrained=False)
        self.target_network = EncoderwithProjection(config,pretrained=False)
        self.online_codebook = Codebook(config)
#         self.target_codebook = target_Codebook(config)

        
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
        l1, q1, p1, e1 = self.online_codebook(self.online_network(view1))
        l2, q2, p2, e2 = self.online_codebook(self.online_network(view2))
        q1 = q1.view(q1.size(0),q1.size(1),-1)
        q2 = q2.view(q2.size(0),q2.size(1),-1)

        # target network forward
        with torch.no_grad():
            if self.use_momentum:
                self._update_target_network(mm)
            else:
                self.target_network=self.online_network
                
            target_z1 = self.target_network(view1)    
            target_z1 = target_z1.detach().clone()
            target_z1 = target_z1.view(target_z1.size(0),target_z1.size(1),-1)

#             tl1, tz1, tp1, te1 = self.target_codebook(target_z1)
            
            
            target_z2 = self.target_network(view2)    
            target_z2 = target_z2.detach().clone()
            target_z2 = target_z2.view(target_z2.size(0),target_z1.size(1),-1)

#             tl2, tz2, tp2, te2 = self.target_codebook(target_z1)
            

#         return l1, q1, p1, e1,l2, q2, p2, e2,tl1, tz1, tp1, te1 ,tl2, tz2, tp2, te2 
        return l1, q1, p1, e1,l2, q2, p2, e2,target_z1 ,target_z2
