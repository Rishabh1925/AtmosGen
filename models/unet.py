import torch
import torch.nn as nn
from .blocks import ConvBlock
from .temporal_encoder import TemporalEncoder
from .time_embedding import TimeEmbedding

class UNet(nn.Module):

    def __init__(self, sequence_length, base_channels=64):
        super().__init__()

        self.temporal = TemporalEncoder(sequence_length, base_channels)

        self.enc1 = ConvBlock(base_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

        self.time_embed = TimeEmbedding(256)
        self.time_mlp = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.time_proj = nn.Linear(256, 512)

    def forward(self, x, t=None):

        if t is not None:
            t_emb = self.time_embed(t)
            t_emb = self.time_mlp(t_emb)
        
        x = self.temporal(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.pool3(e3)
        b = self.bottleneck(b)

        if t is not None:
            t_emb = self.time_embed(t)
            t_emb = self.time_mlp(t_emb)
            t_emb = self.time_proj(t_emb)   
            t_emb = t_emb[:, :, None, None]
            b = b + t_emb

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
    