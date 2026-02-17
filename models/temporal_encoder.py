import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):

    def __init__(self, sequence_length, channels):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=channels,
                kernel_size=(3,3,3),
                padding=1
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x shape: (B, T, C, H, W)

        x = x.permute(0, 2, 1, 3, 4)    # (B, C, T, H, W)
        x = self.conv3d(x)

        # Collapse temporal dimension
        x = torch.mean(x, dim=2)

        return x