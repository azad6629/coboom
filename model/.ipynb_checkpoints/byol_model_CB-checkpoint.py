#-*- coding:utf-8 -*-
import torch
from .basic_modules_CB import EncoderwithProjection, Codebook, Predictor,PredictionModule

# +
class BYOLModel_CB(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.online_network  = EncoderwithProjection(config,pretrained=False)
        self.target_network  = EncoderwithProjection(config,pretrained=False)
#         self.online_codebook = Codebook(config)
#         self.target_codebook = Codebook(config)
        self.predictor       = Predictor(config)
        self.local_predictor = PredictionModule(in_channels=2048, out_channels=2048)

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
        q1_ftr, q1 = self.online_network(view1)
        q2_ftr, q2 = self.online_network(view2)
        q1 = self.predictor(q1)
        q2 = self.predictor(q2)
        q1_ftr = self.local_predictor(q1_ftr)
        q2_ftr = self.local_predictor(q2_ftr)
        
        
#         l1, _, p1, e1 = self.online_codebook(q1_ftr)
#         l2, _, p2, e2 = self.online_codebook(q2_ftr)
        q1_ftr = q1_ftr.view(q1_ftr.size(0),q1_ftr.size(1),-1)
        q2_ftr = q2_ftr.view(q2_ftr.size(0),q2_ftr.size(1),-1)
        

        # target network forward
        with torch.no_grad():
            if self.use_momentum:
                self._update_target_network(mm)
            else:
                self.target_network=self.online_network
                
            z1_ftr, target_z1 = self.target_network(view1)    
            target_z1 = target_z1.detach().clone()
            z1_ftr = z1_ftr.view(z1_ftr.size(0),z1_ftr.size(1),-1)

#             tl1, z1_ftr, tp1, te1 = self.target_codebook(z1_ftr)
            z1_ftr = z1_ftr.detach().clone()
            
            
            z2_ftr, target_z2 = self.target_network(view2)    
            target_z2 = target_z2.detach().clone()
            z2_ftr = z2_ftr.view(z2_ftr.size(0),z2_ftr.size(1),-1)

#             tl2, z2_ftr, tp2, te2 = self.target_codebook(z2_ftr)
            z2_ftr = z2_ftr.detach().clone()
            

#         return l1, q1_ftr, p1, e1, l2, q2_ftr, p2, e2, q1, q2,z1_ftr, target_z1, z2_ftr, target_z2 
        return q1,q1_ftr,q2,q2_ftr, z1_ftr, target_z1, z2_ftr, target_z2 
