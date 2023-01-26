import torch
import torch.nn as nn


class First(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(First, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.Relu(x)
        # x = self.pool(x)
        return x


# Average Pooling or downscale by stride
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.Relu(x)
        # x = self.pool(x)
        return x


class BridgeDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BridgeDown, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class BridgeUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BridgeUp, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.size(dim=0), 256, 4, 8))
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
        return x


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha  # + 1 # to make the mean 1? but we want 0 mean
            #
            # epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x
