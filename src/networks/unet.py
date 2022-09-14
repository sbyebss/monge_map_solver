import torch
import torch.nn.functional as F
from torch import nn

from src.networks.cnn import conv3x3


def compute_cond_module(module, x):
    for m in module:
        x = m(x)
    return x


# pylint: disable=R0902,invalid-name


class Generator(torch.nn.Module):
    def __init__(self, out_channels=3, features=256):
        super().__init__()
        self.act = nn.ReLU()

        self.init_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(features, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ##########################################################
        self.down1 = nn.ModuleList(
            [
                conv3x3(features, features),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.down2 = nn.ModuleList(
            [
                conv3x3(features, features),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.down3 = nn.ModuleList(
            [
                conv3x3(features, features),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.down4 = nn.ModuleList(
            [
                conv3x3(features, features),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        ##########################################################

        self.up1 = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2),
                conv3x3(features, features),
                nn.BatchNorm2d(features, affine=True, track_running_stats=False),
                nn.ReLU(),
            ]
        )
        self.up2 = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2),
                conv3x3(features, features),
                nn.BatchNorm2d(features, affine=True, track_running_stats=False),
                nn.ReLU(),
            ]
        )
        self.up3 = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2),
                conv3x3(features, features),
                nn.BatchNorm2d(features, affine=True, track_running_stats=False),
                nn.ReLU(),
            ]
        )
        self.up4 = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2),
                conv3x3(features, features),
                nn.BatchNorm2d(features, affine=True, track_running_stats=False),
                nn.ReLU(),
            ]
        )

        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.init_layer(x)

        x_1 = compute_cond_module(self.down1, x)
        x_2 = compute_cond_module(self.down2, x_1)
        x_3 = compute_cond_module(self.down3, x_2)
        x_4 = compute_cond_module(self.down4, x_3)

        y_3 = compute_cond_module(self.up1, x_4)
        y_3 = y_3 + x_3

        y_2 = compute_cond_module(self.up2, y_3)
        y_2 = y_2 + x_2

        y_1 = compute_cond_module(self.up3, y_2)
        y_1 = y_1 + x_1

        y = compute_cond_module(self.up4, y_1)
        y = y + x

        output = self.output(y)
        return output


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
