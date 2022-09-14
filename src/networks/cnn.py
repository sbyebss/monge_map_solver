import numpy as np
import torch
from torch import nn


def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)


def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3, features=256, img_size=64):
        super().__init__()
        ##########################################################
        num_downsampling = int(np.log2(img_size / 4)) - 1
        self.down1 = nn.Sequential(
            conv3x3(in_channels, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        down_blocks = [
            nn.Sequential(
                conv3x3(features, features),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            )
            for _ in range(num_downsampling)
        ]
        self.downsampling = nn.Sequential(*down_blocks)

        self.output = nn.Linear(in_features=features * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.downsampling(x)

        x = x.view(x.shape[0], -1)
        output = self.output(x)
        return output
