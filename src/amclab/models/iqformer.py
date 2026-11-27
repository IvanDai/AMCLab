"""IQFormer (WestdoorSad/IQFormer, MIT, Copyright 2024 Shao Mingyuan).
Defaults match main.py for RML2016 and HisarMod2019. For RML2018, L=1024
uses the HisarMod architecture as an explicitly undocumented transfer.
"""
import torch
from torch import nn
from torch.nn import functional as F

class DropPath(nn.Module):
    def __init__(self,p=0.):
        super().__init__()
        self.p=p
    def forward(self,x):
        if not self.training or self.p==0:
            return x
        keep=1-self.p
        mask=x.new_empty((x.shape[0],)+(1,)*(x.ndim-1)).bernoulli_(keep)
        return x*mask.div_(keep)

def _init_weights(m):
    if isinstance(m,nn.Conv1d):
        nn.init.trunc_normal_(m.weight,std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class ConvEncoder_IQ(nn.Module):
    def __init__(self,dim,hidden_dim):
        super().__init__()
        self.dwconv=nn.Conv1d(dim,dim,3,padding=1,groups=dim)
        self.norm=nn.BatchNorm1d(dim)
        self.pwconv1=nn.Conv1d(dim,hidden_dim,1)
        self.act=nn.GELU()
        self.pwconv2=nn.Conv1d(hidden_dim,dim,1)
        self.drop_path=nn.Identity()
        self.layer_scale=nn.Parameter(torch.ones(dim,1))
        self.apply(_init_weights)
    def forward(self,x):
        return x+self.layer_scale*self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))

class LocalRepresentation(ConvEncoder_IQ):
    def __init__(self,dim):
        super().__init__(dim,dim)

class FCN(nn.Module):
    def __init__(self,dim,hidden,drop=0.):
        super().__init__()
        self.norm1=nn.BatchNorm1d(dim)
        self.fc1=nn.Conv1d(dim,hidden,1)
        self.act=nn.GELU()
        self.fc2=nn.Conv1d(hidden,dim,1)
        self.drop=nn.Dropout(drop)
        self.apply(_init_weights)
    def forward(self,x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(self.norm1(x))))))

class EfficientAdditiveAttnetion(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.to_query=nn.Linear(dim,dim)
        self.to_key=nn.Linear(dim,dim)
        self.w_g=nn.Parameter(torch.randn(dim,1))
        self.scale_factor=dim**-.5
        self.Proj=nn.Linear(dim,dim)
        self.final=nn.Linear(dim,dim)
    def forward(self,x):
        q=F.normalize(self.to_query(x),dim=-1)
        k=F.normalize(self.to_key(x),dim=-1)
        a=F.normalize((q@self.w_g)*self.scale_factor,dim=1)
        g=(a*q).sum(dim=1,keepdim=True)
        return self.final(self.Proj(g*k)+q)

class IQFormer_Encoder(nn.Module):
    def __init__(self,dim,mlp_ratio,drop_path,use_layer_scale=False,layer_scale_init_value=1e-5):
        super().__init__()
        self.local_representation=LocalRepresentation(dim)
        self.attn=EfficientAdditiveAttnetion(dim)
        # Stage in upstream does not forward drop_rate to FCN.
        self.linear=FCN(dim,int(dim*mlp_ratio),drop=0.)
        self.drop_path=DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.use_layer_scale=use_layer_scale
        if use_layer_scale:
            self.layer_scale_1=nn.Parameter(layer_scale_init_value*torch.ones(dim,1))
            self.layer_scale_2=nn.Parameter(layer_scale_init_value*torch.ones(dim,1))
    def forward(self,x):
        x=self.local_representation(x)
        y=self.attn(x.transpose(1,2)).transpose(1,2)
        if self.use_layer_scale:
            y=self.layer_scale_1*y
        x=x+self.drop_path(y)
        y=self.linear(x)
        if self.use_layer_scale:
            y=self.layer_scale_2*y
        return x+self.drop_path(y)

class Fusion(nn.Module):
    def __init__(self,channels,drop):
        super().__init__()
        self.Conv=nn.Sequential(nn.Conv1d(channels,channels*2,1),nn.BatchNorm1d(channels*2),
            nn.GELU(),nn.Conv1d(channels*2,channels*2,1))
        self.drop=nn.Dropout(drop)
        self.apply(_init_weights)
    def forward(self,x,stft):
        return self.drop(self.Conv(torch.cat((x,stft),dim=1)))

class Embedding(nn.Module):
    def __init__(self,cin,cout,k,stride,pad):
        super().__init__()
        self.proj=nn.Conv1d(cin,cout,k,stride=stride,padding=pad)
        self.norm=nn.BatchNorm1d(cout)
    def forward(self,x):
        return self.norm(self.proj(x))

def reference_stft(iq):
    # Equivalent to scipy.signal.stft(iq[0], 1., 'blackman', 31, 30, 128)
    # followed by the upstream conversion to a real float32 Tensor.
    signal=iq[:,0].float()
    window=torch.blackman_window(31,periodic=True,device=signal.device,dtype=signal.dtype)
    frames=F.pad(signal,(15,15)).unfold(-1,31,1)
    spec=torch.fft.rfft(frames*window,n=128,dim=-1)/window.sum()
    return spec.real[...,:32].transpose(1,2).unsqueeze(1).contiguous().to(iq.dtype)

class IQFormer(nn.Module):
    def __init__(self,num_classes=11,seq_length=128,layers=None,embed_dims=(64,64,64),
                 mlp_ratio=4.,down_patch_size=3,down_stride=2,down_pad=1,
                 drop_rate=.2,drop_path_rate=None,use_layer_scale=False,
                 layer_scale_init_value=1e-5,stft_freq_bins=32,vit_num=1):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        self.seq_length = int(seq_length)
        if layers is None:
            layers=(2,3,2) if seq_length==128 else (3,3,3)
        if drop_path_rate is None:
            drop_path_rate=0. if seq_length==128 else .2
        if stft_freq_bins!=32:
            raise ValueError("The reference STFT has 32 frequency bins.")
        if len(layers)!=len(embed_dims) or embed_dims[0]%16:
            raise ValueError("Stage depths/dimensions must align and first dim must be divisible by 16.")
        self.layers=tuple(layers)
        self.embed_dims=tuple(embed_dims)
        self.BN=nn.BatchNorm1d(2)
        self.BN_stft=nn.BatchNorm2d(1)
        c=embed_dims[0]//8
        self.patch_embedIQ=nn.Sequential(nn.Conv1d(2,c,5,padding=2,groups=2),nn.BatchNorm1d(c))
        self.patch_embedSTFT=nn.Sequential(nn.Conv2d(1,c,(32,1)),nn.BatchNorm2d(c),nn.ReLU())
        self.fusion=Fusion(embed_dims[0]//4,drop_rate)
        network=[]
        for i,(depth,dim) in enumerate(zip(layers,embed_dims)):
            stage=[]
            for j in range(depth):
                dpr=drop_path_rate*(sum(layers[:i])+j)/max(sum(layers)-1,1)
                if depth-j<=vit_num:
                    stage.append(IQFormer_Encoder(dim,mlp_ratio,dpr,use_layer_scale,layer_scale_init_value))
                else:
                    stage.append(ConvEncoder_IQ(dim,int(dim*mlp_ratio)))
            network.append(nn.Sequential(*stage))
            if i<len(layers)-1 and dim!=embed_dims[i+1]:
                network.append(Embedding(dim,embed_dims[i+1],down_patch_size,down_stride,down_pad))
        self.network=nn.ModuleList(network)
        d=embed_dims[0]//2
        self.patch_LSTM=nn.LSTM(d,d,bidirectional=True,batch_first=True,num_layers=2,dropout=drop_rate)
        self.norm=nn.BatchNorm1d(embed_dims[-1])
        self.head=nn.Linear(embed_dims[-1],num_classes)
        self.apply(_init_weights)
        self.globalavgpool=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.globalmaxpool=nn.Sequential(nn.AdaptiveMaxPool1d(1),nn.Flatten())
    def _compute_stft(self,iq):
        return reference_stft(iq)
    def forward(self,x,stft=None):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        if stft is None:
            stft=self._compute_stft(x)
        if stft.shape!=(x.shape[0],1,32,self.seq_length):
            raise ValueError("STFT must have shape [B,1,32,L].")
        a=self.patch_embedIQ(self.BN(x))
        b=self.patch_embedSTFT(self.BN_stft(stft)).squeeze(2)
        h,_=self.patch_LSTM(self.fusion(a,b).transpose(1,2))
        h=h.transpose(1,2)
        for block in self.network:
            h=block(h)
        return self.head(self.globalavgpool(self.norm(h)))

def iqformer_base(num_classes=11,seq_length=128,**kwargs):
    return IQFormer(num_classes,seq_length,**kwargs)
def iqformer_small(num_classes=11,seq_length=128,**kwargs):
    return IQFormer(num_classes,seq_length,layers=(1,1,1),embed_dims=(32,32,32),**kwargs)
def iqformer_large(num_classes=11,seq_length=128,**kwargs):
    return IQFormer(num_classes,seq_length,layers=(2,3,2),embed_dims=(64,96,128),**kwargs)
