"""MCformer port for AMCLab, based on CSRR implementation.

Source:
https://github.com/Singingkettle/ChangShuoRadioRecognition/blob/main/csrr/models/backbones/mcformer.py
Config:
https://github.com/Singingkettle/ChangShuoRadioRecognition/blob/main/configs/mcformer/mcformer_iq-shape-F-L-deepsig-201610A.py

The backbone computation is kept faithful to CSRR:
Conv1d(2->32,k=65,padding='same') -> ReLU ->
4-layer TransformerEncoder (d_model=32,nhead=4,dim_feedforward=32) ->
take first 4 sequence positions -> flatten 128 ->
Linear(128,128) -> ReLU -> Dropout(0.5) -> Linear(128,C).

Input/output adapted for AMCLab:
[B,2,L] -> [B,C] logits.
No positional encoding is added because the referenced implementation does not
define one. The squeeze(dim=2) in CSRR is a no-op for L>1; omitted here without
changing tensor values.
"""
import torch
from torch import nn


class MCFormer(nn.Module):
    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        fea_dim: int = 128,
        frame_length=None,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MCFormer arguments: {unknown}")
        if frame_length is not None:
            if seq_length != 128 and seq_length != frame_length:
                raise ValueError("seq_length and frame_length disagree")
            seq_length = frame_length
        if seq_length < 4:
            raise ValueError("MCFormer requires seq_length >= 4")

        self.seq_length = int(seq_length)
        self.fea_dim = int(fea_dim)
        self.num_classes = int(num_classes)

        self.cnn = nn.Sequential(
            nn.Conv1d(2, self.fea_dim, kernel_size=65, padding="same"),
            nn.SELU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fea_dim,
            nhead=4,
            dim_feedforward=self.fea_dim,
            batch_first=True,
        )
        self.tnn = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Sequential(
            nn.Linear(4 * self.fea_dim, 128),
            nn.SELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input [B, 2, L], got {tuple(x.shape)}")
        if x.shape[2] != self.seq_length:
            raise ValueError(
                f"Model configured for L={self.seq_length}, got L={x.shape[2]}"
            )

        x = self.cnn(x)                 # [B, 32, L]
        x = x.transpose(1, 2)           # [B, L, 32]
        x = self.tnn(x)                 # [B, L, 32]
        x = x[:, :4, :]                 # [B, 4, 32], faithful to CSRR
        x = x.reshape(x.shape[0], 4 * self.fea_dim)
        return self.classifier(x)       # [B, num_classes] logits


# AMCLab model registry compatibility.
MCformer = MCFormer
