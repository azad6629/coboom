# -*- coding: utf-8 -*-
import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_image,self.next_input_ids,self.next_attention_mask,self.next_rad = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_input_ids= None
            self.next_attention_mask= None
            self.next_rad=None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_input_ids = self.next_input_ids.cuda(non_blocking=True)
            self.next_attention_mask = self.next_attention_mask.cuda(non_blocking=True)
            self.next_rad = self.next_rad.cuda(non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_image
        input_ids = self.next_input_ids
        attention_mask = self.next_attention_mask
        rad_features = self.next_rad
        
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if input_ids is not None:
            input_ids.record_stream(torch.cuda.current_stream())
        if attention_mask is not None:
            attention_mask.record_stream(torch.cuda.current_stream())
        if rad_features is not None:
            rad_features.record_stream(torch.cuda.current_stream())
        
        self.preload()
        return input,input_ids,attention_mask,rad_features
