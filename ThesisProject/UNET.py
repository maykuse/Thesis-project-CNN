from UNET_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down1 = Down(in_channels, 64)
        # self.first = First(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.bridge1 = BridgeDown(8192, 512)
        self.bridge2 = BridgeUp(512, 8192)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.last = Last(64, out_channels)

    def forward(self, x):
        x1 = self.down1(x)  # 64
        # x1 = self.first(x)  # 64
        x2 = self.down2(x1)  # 128
        x3 = self.down3(x2)  # 256
        x4 = self.bridge1(x3)  # 1x8192
        x5 = self.bridge2(x4)  # 256x4x8
        x6 = self.up1(x5, x2)  # 128-128
        x7 = self.up2(x6, x1)  # 64-64
        x8 = self.last(x7)  # 64 --> 7
        return x8
