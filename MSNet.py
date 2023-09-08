""" Full assembly of the parts to form the complete network """
from torch import nn
import torch
from torch.distributions import Normal, Independent
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1) 

    
# class Dropout(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
    
#         self.dropout = nn.Dropout(p = 0.5)

#     def forward(self, x):
#         x = self.dropout(x)
#         return x

# 用于初始化网络参数的函数
def weights_init(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.normal_(module.weight, 0, np.sqrt(1/module.in_channels)) #初始化每个卷积权重0~1/2 0~1/48
        #torch.nn.init.normal_(module.weight, 0, 0)
        #print('module.in_channels:', module.in_channels) 2 48 48 48 48 48 48 48
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
class MSNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        # input_tensor的参数
        self.r = 3 ** 0.5
        self.size = 256
        self.coord_range = torch.linspace(-self.r, self.r, self.size)
        self.x = self.coord_range.view(-1, 1).repeat(1, self.coord_range.size(0))
        self.y = self.coord_range.view(1, -1).repeat(self.coord_range.size(0), 1)
        
        # 网络的参数
        self.num_output_channels = 1             #
        self.layers = []                         # 用于网络扩张
        self.kernel_size = 1                     # 神经网络卷积核大小
        self.num_layers  = 4                     # 网络层数
        self.num_hidden_channels = 16            # 中间层的大小 2-->24-->actv 48--->24--->actv 48--->......--->6
        self.activation_fn = CompositeActivation # 激活函数
        #self.activation_fn = torch.nn.ReLU # 激活函数
        self.normalize = False                   # 默认不使用归一化
        
#         self.dropout = Dropout
        
        # 定义神经网络
        for i in range(self.num_layers):
            self.out_c = self.num_hidden_channels
            self.in_c = self.out_c * 2 # * 2 for composite activation
            #self.in_c = self.out_c  # * 2 for composite activation
            if i == 0:
                self.in_c = 2 # 因为input_tensor--->torch.Size([batch, 2, size, size])
            if i ==self. num_layers - 1:
                self.out_c = self.num_output_channels
            self.layers.append(('conv{}'.format(i), torch.nn.Conv2d(self.in_c, self.out_c, self.kernel_size)))
            if self.normalize:
                self.layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(self.out_c)))
            if i < self.num_layers - 1:
                self.layers.append(('actv{}'.format(i), self.activation_fn()))
                # 加入dropout
                # self.layers.append(('dropout{}'.format(i), self.dropout()))
            else:
                self.layers.append(('output', torch.nn.Sigmoid())) 
                
        # 初始化的模型
        self.net = torch.nn.Sequential(OrderedDict(self.layers))
        
        # 初始化网络参数
        self.net.apply(weights_init)
        
        # 将最后一个Conv2D的权重置为0
        torch.nn.init.zeros_(dict(self.net.named_children())['conv{}'.format(self.num_layers - 1)].weight)

    def forward(self, seg):
        batch = seg.size(0)

        # 因为batch参数，input_tensor的初始化放到下面
        # 其实x y都一样，初始化的input_tensor基本都是--->torch.Size([8, 2, 256, 256])
        input_tensor = torch.stack([self.x, self.y], dim=0).unsqueeze(0).repeat(batch,1,1,1).cuda()
        #print(input_tensor.shape)
        # 生成加权图谱
        final_seg = self.net(input_tensor)
#         print("\nfinal_seg.shape:", final_seg.shape ) # final_seg.shape: torch.Size([1, 1, 256, 256])
        
        # Sigmoid
        # [8,6,256,256] [8,6,256,256]
#         maps = torch.nn.Softmax(dim = 1)(maps)
#         final_seg = torch.multiply(seg,maps).sum(dim = 1, keepdim = False) # [8, 256, 256]
        
#         print("\nmaps max:", torch.max(maps[0,:,:,:]) ) #maps max: tensor(0.1667, device='cuda:0', grad_fn=<MaxBackward1>)
#         print("\nmaps min:", torch.min(maps[0,:,:,:]) ) #maps min: tensor(0.1667, device='cuda:0', grad_fn=<MinBackward1>)
        #print("\nfinal_seg:", torch.max(final_seg[0,:,:]) ) # max = 1
        
        return final_seg
