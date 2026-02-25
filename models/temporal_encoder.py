import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):

    def __init__(self, base_channels):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.conv3d(3, base_channels, kernel_sizes=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = self.conv3d(x)
        x = torch.mean(x, dim=2)
        return x