import torch
import torch.nn as nn


def gaussian_dropout(data: torch.Tensor, p: torch.Tensor):
    """
    Function for applying parametric Gaussian dropout
    Parameters:
        data: input data, expected shape (batch_size, num_channels, h, w)
        p: dropout rates in [0, 1], expected shape (batch_size, num_channels)
    Returns:
        out: Gaussian dropout output, shape (batch_size, num_channels, h, w)
    """
    p = p.view(*p.shape, 1, 1)
    alpha = p / (1. - p)
    noise = torch.randn_like(data)
    weights = 1. + torch.sqrt(alpha) * noise
    "weights = theta + theta * sqrt(alpha), with fixed theta = 1"
    out = weights * data
    return out


def gaussian_dropout_image(data: torch.Tensor, p: torch.Tensor):
    """
        The only difference: p is given as 3D to keep track of the gradient of pixels
        p: dropout rates in [0, 1], expected shape (batch_size, num_channels, h, w)
    """
    alpha = p / (1. - p)
    noise = torch.randn_like(data)
    weights = 1. + torch.sqrt(alpha) * noise
    out = weights * data
    return out


# Dimension reduction can either be done with stride=2 or average pooling after convolution.
# Here stride = 2 is defined to reduce the resolution by half.
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.Relu(x)
        # x = self.pool(x)
        return x


# Features are flattened to 1D vector
class BottomDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottomDown, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


# 1D Features are reshaped into an appropriate resolution to forward it to the decoding path
class BottomUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottomUp, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.size(dim=0), 256, 4, 8))
        return x


# Deconvolution to double the resolution, then...
# Concatenation of the features from the encoding path to reduce information loss
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


# Last layer reduces the final resolution to the desired number of outputs to be predicted
class Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Last, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.up(x)
        x = self.Relu(x)
        x = self.conv(x)
        return x


