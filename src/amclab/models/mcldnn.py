"""Paper-faithful PyTorch MCLDNN for AMCLab."""
import torch
from torch import nn


class MCLDNN(nn.Module):
    def __init__(self, num_classes: int = 11, seq_length: int = 128, dropout: float = 0.5):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError(f"MCLDNN supports seq_length 128 or 1024, got {seq_length}.")
        self.seq_length = int(seq_length)

        # Keras Conv2D(kernel=(2,8), padding="same"):
        # height padding=1 (bottom only), time padding=7 (left=3/right=4).
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 1)),
            nn.Conv2d(1, 50, kernel_size=(2, 8)),
            nn.ReLU(inplace=True),
        )
        # Keras Conv1D(kernel=8, padding="causal").
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0.0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0.0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(inplace=True),
        )
        # Keras Conv2D(kernel=(1,8), padding="same").
        self.conv4 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(50, 50, kernel_size=(1, 8)),
            nn.ReLU(inplace=True),
        )
        # Keras Conv2D(kernel=(2,5), padding="valid").
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.SELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.SELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.reset_parameters_keras_like()

    def reset_parameters_keras_like(self) -> None:
        # Keras Conv/Dense defaults: Glorot uniform.
        # CuDNNLSTM: Glorot input weights, orthogonal recurrent weights,
        # and unit forget-gate bias.
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
                with torch.no_grad():
                    # PyTorch gate order: input, forget, cell, output.
                    param[128:256].fill_(1.0)
            elif "bias_hh" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected input [B, 2, {self.seq_length}], got {tuple(x.shape)}")

        # [B, 2, L] -> [B, 1, 2, L]
        iq = x.unsqueeze(1)
        joint = self.conv1(iq)                         # [B, 50, 2, L]
        i_feat = self.conv2(iq[:, :, 0, :])           # [B, 50, L]
        q_feat = self.conv3(iq[:, :, 1, :])           # [B, 50, L]
        separate = self.conv4(torch.stack((i_feat, q_feat), dim=2))
        fused = self.conv5(torch.cat((joint, separate), dim=1))  # [B, 100, 1, L-4]

        # Critical PyTorch NCHW -> [B, time, channels] conversion.
        # A plain reshape here scrambles the time/channel ordering.
        sequence = fused.squeeze(2).transpose(1, 2).contiguous()  # [B, L-4, 100]
        sequence, _ = self.lstm(sequence)
        return self.classifier(sequence[:, -1, :])  # logits
