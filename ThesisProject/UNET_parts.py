import torch
import torch.nn as nn


class First(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(First, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.Relu(x)
        x = self.pool(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.Relu(x)
        x = self.pool(x)
        return x


class Bridge(nn.Module):
    # what should the kernel size be such that the output of this layer is 1x512
    def __init__(self, in_channels, out_channels):  # parameter input length?
        super(Bridge, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, 7, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.Relu(x)
        x = self.pool(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.Relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.Relu(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.Relu(x)
        return x


class Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Last, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.Relu = nn.ReLU()
        self.Tan = nn.Tanh()

    def forward(self, x):
        x = self.up(x)
        x = self.Relu(x)
        x = self.conv(x)
        x = self.Tan(x)
        return x
