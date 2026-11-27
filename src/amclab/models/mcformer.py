"""MCFormer: convolution, four BERT-style encoders, first-four-token head.

L=128: released MCformerLarge configuration (32 channels, kernel=65).
L=1024: AMCLab budget-matched adaptation (48 channels, kernel=377);
the original publication and IQFormer repository do not establish
this long-sequence preset. Explicit width/kernel overrides are supported.
"""
from torch import nn

class SelfAttentionTransformerBERT(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_prob, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4), nn.GELU(),
            nn.Linear(hidden_size*4, hidden_size), nn.Dropout(dropout_prob),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        y, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class MCFormer(nn.Module):
    def __init__(self, num_classes=11, seq_length=128, hidden_size=None,
                 kernel_size=None, num_heads=4):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        self.seq_length = seq_length
        hidden_size = (32 if seq_length == 128 else 48) if hidden_size is None else hidden_size
        kernel_size = (65 if seq_length == 128 else 377) if kernel_size is None else kernel_size
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be a positive multiple of num_heads.")
        if not 1 <= kernel_size <= seq_length - 1:
            raise ValueError("Need at least four post-convolution positions.")
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        # Conv1d is equivalent to the source Conv2d with width=1.
        self.conv_layer = nn.Conv1d(2, hidden_size, kernel_size, padding=1)
        self.selu = nn.SELU()
        self.transformer_layers = nn.Sequential(*(
            SelfAttentionTransformerBERT(hidden_size, num_heads) for _ in range(4)
        ))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4*hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        x = self.selu(self.conv_layer(x)).transpose(1, 2)
        x = self.transformer_layers(x)
        x = self.flatten(x[:, :4, :].transpose(1, 2))
        return self.fc2(self.selu(self.fc1(x)))


MCformer = MCFormer
