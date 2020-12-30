import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(
            3, 3), bias=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(
            3, 3), bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool_conv(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.double_conv = DoubleConv(
            in_channels, out_channels, in_channels // 2)
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        self.center_crop = CenterCrop(x1.shape[-2:])
        x2 = self.center_crop(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.double_conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x
