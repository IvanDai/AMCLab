"""AMC_Net from WestdoorSad/IQFormer/model/AMCNET.py (MIT, Shao Mingyuan 2024).

The 1024 constructor is the documented default. 128 substitutes sig_len=128
with the same remaining defaults; no verified separate 128 training config.
The upstream deepcopy of a tensor is replaced by a non-mutated reference
(same values/gradients, avoids errors on non-leaf tensors).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv_block=nn.Sequential(nn.ZeroPad2d((1,1,0,0)),
            nn.Conv2d(in_channel,out_channel,(1,3)),nn.ReLU(inplace=True),nn.BatchNorm2d(out_channel))
    def forward(self,x):
        return self.conv_block(x)

class MultiScaleModule(nn.Module):
    def __init__(self,out_channel):
        super().__init__()
        def branch(k):
            return nn.Sequential(nn.ZeroPad2d((k//2,k//2,0,0)),
                nn.Conv2d(1,out_channel//3,(2,k)),nn.ReLU(inplace=True),nn.BatchNorm2d(out_channel//3))
        self.conv_3,self.conv_5,self.conv_7=branch(3),branch(5),branch(7)
    def forward(self,x):
        return torch.cat((self.conv_3(x),self.conv_5(x),self.conv_7(x)),dim=1)

class TinyMLP(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(N,N//4),nn.ReLU(inplace=True),nn.Linear(N//4,N),nn.Tanh())
    def forward(self,x):
        return self.mlp(x)

class AdaCorrModule(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.Im=TinyMLP(N)
        self.Re=TinyMLP(N)
    def forward(self,x):
        # FFT in float32 avoids half precision FFT restrictions under AMP.
        z=torch.fft.fft(x.float() if x.dtype in (torch.float16,torch.bfloat16) else x,dim=-1)
        re=self.Re(z.real)*z.real
        im=self.Im(z.imag)*z.imag
        correction=torch.fft.ifft(torch.complex(re.float() if re.dtype in (torch.float16,torch.bfloat16) else re,
            im.float() if im.dtype in (torch.float16,torch.bfloat16) else im),dim=-1).real
        return x+correction.to(x.dtype)

class FeaFusionModule(nn.Module):
    def __init__(self,num_heads,input_size,hidden_size):
        super().__init__()
        if hidden_size%num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.num_attention_heads=num_heads
        self.attention_head_size=hidden_size//num_heads
        self.key_layer=nn.Linear(input_size,hidden_size)
        self.query_layer=nn.Linear(input_size,hidden_size)
        self.value_layer=nn.Linear(input_size,hidden_size)
        self.dropout=nn.Dropout(.5)
    def trans_to_multiple_heads(self,x):
        return x.reshape(*x.shape[:-1],self.num_attention_heads,self.attention_head_size).permute(0,2,1,3)
    def forward(self,x):
        k=self.trans_to_multiple_heads(self.key_layer(x))
        q=self.trans_to_multiple_heads(self.query_layer(x))
        v=self.trans_to_multiple_heads(self.value_layer(x))
        a=self.dropout((q@k.transpose(-1,-2)/math.sqrt(self.attention_head_size)).softmax(-1))
        context=a@v
        # Intentional upstream layout: [B, heads*channels, head_size].
        return context.contiguous().reshape(x.shape[0],-1,self.attention_head_size)

class AMCNet(nn.Module):
    def __init__(self,num_classes=11,seq_length=128,extend_channel=36,latent_dim=512,
                 num_heads=2,conv_chan_list=None):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        self.seq_length = int(seq_length)
        channels=list(conv_chan_list or [36,64,128,256])
        if extend_channel%3 or channels[0]!=extend_channel or latent_dim!=num_heads*channels[-1]:
            raise ValueError("MSM/stem/attention dimensions must match latent_dim.")
        self.ACM=AdaCorrModule(seq_length)
        self.MSM=MultiScaleModule(extend_channel)
        self.FFM=FeaFusionModule(num_heads,seq_length,seq_length)
        self.Conv_stem=nn.Sequential()
        for i,(a,b) in enumerate(zip(channels[:-1],channels[1:])):
            self.Conv_stem.add_module(f"conv_stem_{i}",Conv_Block(a,b))
        self.GAP=nn.AdaptiveAvgPool1d(1)
        self.classifier=nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.Dropout(.5),
            nn.PReLU(),nn.Linear(latent_dim,num_classes))
    def forward(self,x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        x=self.ACM(x.unsqueeze(1))
        # Guard only the undefined all-zero input case in the upstream.
        x=x/x.norm(p=2,dim=-1,keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
        x=self.Conv_stem(self.MSM(x)).squeeze(2)
        return self.classifier(self.GAP(self.FFM(x)).squeeze(2))

AMC_Net=AMCNet
