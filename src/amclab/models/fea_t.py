"""FEA-T, merged defaults from WestdoorSad/IQFormer FEA_T128.py and FEA_T1024.py.
Source credits: Yantao Chen; IQFormer repository MIT (Shao Mingyuan, 2024).
Default path uses DB-GLU-add, no talking heads or RealFormer.
"""
import torch
from torch import nn

class _MHSABlock(nn.Module):
    def __init__(self,dim,head_dim,heads,dropout):
        super().__init__()
        self.heads,self.head_dim=heads,head_dim
        self.scale=head_dim**.5
        self.to_q=nn.Linear(dim,heads*head_dim,bias=False)
        self.to_k=nn.Linear(dim,heads*head_dim,bias=False)
        self.to_v=nn.Linear(dim,heads*head_dim,bias=False)
        self.proj_v=nn.Linear(heads*head_dim,dim,bias=False)
        self.Dp_attn=nn.Dropout(dropout)
        self.Dp_v=nn.Dropout(dropout)
    def forward(self,x):
        b,p,_=x.shape
        def split(y):
            return y.reshape(b,p,self.heads,self.head_dim).transpose(1,2)
        q=split(self.to_q(x))/self.scale
        k=split(self.to_k(x))
        v=split(self.to_v(x))
        a=self.Dp_attn((q@k.transpose(-1,-2)).softmax(-1))
        return self.Dp_v(self.proj_v((a@v).transpose(1,2).reshape(b,p,-1)))

class _DBGLU(nn.Module):
    def __init__(self,dim,hidden,dropout):
        super().__init__()
        if hidden%2:
            raise ValueError("DB-GLU hidden size must be even.")
        self.linear1=nn.Linear(dim,hidden)
        self.linear2=nn.Linear(hidden//2,dim)
        self.Act=nn.GELU()
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        a,b=self.linear1(x).chunk(2,dim=-1)
        return self.linear2(self.dropout(a*self.Act(b)+b*self.Act(a)))

class _TransformerLayer(nn.Module):
    def __init__(self,dim,head_dim,hidden,heads,dropout):
        super().__init__()
        self.mhsa=_MHSABlock(dim,head_dim,heads,dropout)
        self.ffn=_DBGLU(dim,hidden,dropout)
        self.norm_1=nn.LayerNorm(dim,eps=1e-5)
        self.norm_2=nn.LayerNorm(dim,eps=1e-5)
    def forward(self,x):
        x=self.norm_1(x+self.mhsa(x))
        return self.norm_2(x+self.ffn(x))

class _Frame(nn.Module):
    def __init__(self,patch_size,stride):
        super().__init__()
        self.PatchSize,self.Stride=patch_size,stride
        self.Embeddinbg=nn.Linear(patch_size*2,patch_size*2)
    def forward(self,x):
        # Vectorized upstream loop: [I frame, Q frame] (not interleaved).
        b=x.shape[0]
        frames=x.unfold(-1,self.PatchSize,self.Stride)
        frames=frames.permute(0,2,1,3).reshape(b,-1,2*self.PatchSize)
        return self.Embeddinbg(frames)

class FEAT(nn.Module):
    def __init__(self,num_classes=11,seq_length=128,patch_size=None,overlap=None,
                 layer_num=8,num_heads=4,d_head=16,hidden_features=256,dropout=0.):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        self.seq_length = int(seq_length)
        patch_size=(16 if seq_length==128 else 32) if patch_size is None else patch_size
        overlap=(.2 if seq_length==128 else .5) if overlap is None else overlap
        self.patch_size=patch_size
        self.stride=int(patch_size*overlap)
        if self.stride<1 or patch_size>seq_length:
            raise ValueError("Invalid frame size/stride.")
        self.frame_num=(seq_length-patch_size)//self.stride+1
        d=patch_size*2
        self.cls_token=nn.Parameter(torch.zeros(1,1,d))
        self.pos_embed=nn.Parameter(torch.zeros(1,self.frame_num+1,d))
        self.Embedding=_Frame(patch_size,self.stride)
        self.Enc=nn.ModuleList([_TransformerLayer(d,d_head,hidden_features,num_heads,dropout)
                                for _ in range(layer_num)])
        self.classifier=nn.Linear(d,num_classes)
        nn.init.trunc_normal_(self.cls_token,std=.02)
        nn.init.trunc_normal_(self.pos_embed,std=.02)
    def forward(self,x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        x=self.Embedding(x)
        x=torch.cat((x,self.cls_token.expand(x.shape[0],-1,-1)),dim=1)+self.pos_embed
        for enc in self.Enc:
            x=enc(x)
        return self.classifier(x[:,-1,:])

FEA_T=FEAT
def fea_t_base(num_classes=11,seq_length=128,**kwargs):
    return FEAT(num_classes,seq_length,**kwargs)
def fea_t_small(num_classes=11,seq_length=128,**kwargs):
    return FEAT(num_classes,seq_length,layer_num=4,hidden_features=128,**kwargs)
