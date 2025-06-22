import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from skimage.feature import canny
import numpy as np


class DeformableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, 3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(16, out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(16, channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = self.norm2(self.conv2(x))
        return x + residual


class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn


class EnhancedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(4, 64, 4, 2, 1)
        self.enc2 = DeformableBlock(64, 128)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.res_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(6)])
        self.attn = AttentionModule(256)
        self.dec3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = DeformableBlock(128, 64)
        self.dec1 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = F.leaky_relu(self.enc1(x), 0.2)
        e2 = self.enc2(e1)
        e3 = F.leaky_relu(self.enc3(e2), 0.2)
        r = e3
        for block in self.res_blocks:
            r = block(r)
        r = self.attn(r)
        d3 = F.leaky_relu(self.dec3(r), 0.2)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return self.tanh(d1)


def get_edge_map(image_tensor):
    image = image_tensor.detach().cpu()
    img_np = image.numpy().transpose(1, 2, 0)
    img_np = (img_np + 1) / 2
    edge = canny(img_np.mean(axis=2), sigma=2)
    return torch.tensor(edge, dtype=torch.float32).unsqueeze(0)
