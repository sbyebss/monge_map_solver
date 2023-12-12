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


# pylint: skip-file

Half_width = 512
layer_width = 128

# Best model from the leadboard:
# https://paperswithcode.com/sota/image-classification-on-kuzushiji-mnist


class SpinalNet(nn.Module):
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """

    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def __init__(self, num_class=26, **kwargs):
        super().__init__()
        self.l1 = self.two_conv_pool(3, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(layer_width * 4, num_class),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(
            torch.cat([x[:, Half_width : 2 * Half_width], x1], dim=1)
        )
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(
            torch.cat([x[:, Half_width : 2 * Half_width], x3], dim=1)
        )

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x
