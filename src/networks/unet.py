import torch
import torch.nn.functional as F
from torch import nn

from src.networks.cnn import conv3x3


def compute_cond_module(module, x):
    for m in module:
        x = m(x)
    return x


# pylint: disable=R0902,invalid-name, unused-argument


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


class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_factor=32, bilinear=True, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_factor = base_factor

        self.inc = DoubleConv(n_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor, bilinear)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor, bilinear)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor, bilinear)
        self.up4 = Up(2 * base_factor, base_factor, bilinear)
        self.outc = OutConv(base_factor, n_classes)

    def forward(self, x, *args):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
