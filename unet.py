from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
#from fp16_util import convert_module_to_f16, convert_module_to_f32
from nn import (
#     checkpoint,
#     conv_nd,
    linear,
#     avg_pool_nd,
#     zero_module,
#     normalization,
    timestep_embedding,
)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    """

    def __init__(
        self,
        model_channels,
        dropout
#         image_size,
#         in_channels,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.dropout = dropout
        
        time_embed_dim = 256
        
        layers_list = [32,48,48]
        
        
        self.time_embed = th.nn.Parameter(th.FloatTensor(6,time_embed_dim), requires_grad=True)       
        
        
        ###
        ###
        ###
        self.up = nn.Sequential(nn.ConvTranspose2d(256, 32, 2, stride=2),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),)
        
        
        
        ###
        ###
        ###
        self.emb_firstconv = nn.Sequential(nn.SiLU(),
                                           linear(time_embed_dim,32),
                                          )
        self.firstconv_cup = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),)
        self.firstconv_disc = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),)

       
        ###
        ###
        ###
        self.emb_layers_00 = nn.Sequential(nn.SiLU(),
                                           linear(time_embed_dim,layers_list[0]),
                                          )
        self.out_layers_00_cup = nn.Sequential(nn.BatchNorm2d(layers_list[0]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[0],layers_list[0], 3, padding=1),
                                        nn.BatchNorm2d(layers_list[0]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[0],layers_list[1], 3, padding=1),)
        self.out_layers_00_disc = nn.Sequential(nn.BatchNorm2d(layers_list[0]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[0],layers_list[0], 3, padding=1),
                                        nn.BatchNorm2d(layers_list[0]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[0],layers_list[1], 3, padding=1),)
        
        ###
        ###
        ###
        self.emb_layers_01 = nn.Sequential(nn.SiLU(),
                                           linear(time_embed_dim,layers_list[1]),
                                          )
        self.out_layers_01_cup = nn.Sequential(nn.BatchNorm2d(layers_list[1]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[1],layers_list[1], 3, padding=1),
                                        nn.BatchNorm2d(layers_list[1]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[1],layers_list[2], 3, padding=1),)
        self.out_layers_01_disc = nn.Sequential(nn.BatchNorm2d(layers_list[1]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[1],layers_list[1], 3, padding=1),
                                        nn.BatchNorm2d(layers_list[1]),
                                        nn.SiLU(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Conv2d(layers_list[1],layers_list[2], 3, padding=1),)
        
        
        ###
        ###
        ###
        self.emb_layers_last = nn.Sequential(nn.SiLU(),
                                           linear(time_embed_dim,layers_list[2]),
                                          )
        self.out_cup = nn.Sequential(nn.BatchNorm2d(layers_list[2]),
                                      nn.SiLU(),
                                      nn.Conv2d(layers_list[2],layers_list[2], 3, padding=1),
                                      nn.BatchNorm2d(layers_list[2]),
                                      nn.SiLU(),
                                      nn.Conv2d(layers_list[2],2, 1, padding=0),
                                     )
        self.out_disc = nn.Sequential(nn.BatchNorm2d(layers_list[2]),
                                      nn.SiLU(),
                                      nn.Conv2d(layers_list[2],layers_list[2], 3, padding=1),
                                      nn.BatchNorm2d(layers_list[2]),
                                      nn.SiLU(),
                                      nn.Conv2d(layers_list[2],2, 1, padding=0),
                                     )
      
        
        
    def forward(self, x, T):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        x = self.up(x)
        
        emb = self.time_embed[T[0]:T[0]+1,:]
        for index in T[0:-1]:
            emb = th.cat((emb,self.time_embed[index:index+1,:]), dim=0)
        

        emb_out = self.emb_firstconv(emb)
        
        x_cup = x + emb_out[..., None, None]
        x_cup = self.firstconv_cup(x_cup)        
        x_disc = x + emb_out[..., None, None]
        x_disc = self.firstconv_disc(x_disc)        
        
        emb_out = self.emb_layers_00(emb)
        
        x_cup = x_cup + emb_out[..., None, None]
        x_cup = self.out_layers_00_cup(x_cup)
        x_disc = x_disc + emb_out[..., None, None]
        x_disc = self.out_layers_00_disc(x_disc)

        emb_out = self.emb_layers_01(emb)
        
        x_cup = x_cup + emb_out[..., None, None]
        x_cup = self.out_layers_01_cup(x_cup)
        x_disc = x_disc + emb_out[..., None, None]
        x_disc = self.out_layers_01_disc(x_disc)
        
        emb_out = self.emb_layers_last(emb)
        
        x_cup = x_cup + emb_out[..., None, None]
        out_cup = self.out_cup(x_cup)
        x_disc = x_disc + emb_out[..., None, None]
        out_disc = self.out_disc(x_disc)       
        
        return out_cup, out_disc






