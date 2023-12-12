from typing import Optional

import torch
from jamtorch.data import get_batch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

__all__ = ["logit_transform", "data_transform", "inverse_data_transform"]


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, x):
    if config.uniform_dequantization:
        x = x / 256.0 * 255.0 + torch.rand_like(x) / 256.0
    if config.gaussian_dequantization:
        x = x + torch.randn_like(x) * 0.01

    if config.rescaled:
        x = 2 * x - 1.0
    elif config.logit_transform:
        x = logit_transform(x)

    if config.image_mean is not None and config.image_std is not None:
        return (
            x - torch.FloatTensor(config.image_mean).to(x.device)[:, None, None]
        ) / torch.FloatTensor(config.image_std).to(x.device)[:, None, None]
    return x


def inverse_data_transform(config, x):
    if config.image_mean is not None and config.image_std is not None:
        x = (
            x * torch.FloatTensor(config.image_std).to(x.device)[:, None, None]
            + torch.FloatTensor(config.image_mean).to(x.device)[:, None, None]
        )

    if config.logit_transform:
        x = torch.sigmoid(x)
    elif config.rescaled:
        x = (x + 1.0) / 2.0

    return x


# pylint: disable=W0223
class ImageModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg

    def data_transform(self, feed_dict):
        if isinstance(feed_dict, list):
            feed_dict = feed_dict[0]
        feed_dict = feed_dict.float()
        return data_transform(self.cfg, feed_dict)

    def inverse_data_transform(self, x):
        return inverse_data_transform(self.cfg, x)


class InpaintImageModule(ImageModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_train_source: Optional[Dataset] = None
        self.data_train_target: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        if self.cfg.dataset == "CELEBA":
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(140),
                    transforms.Resize(self.cfg.image_size),
                    transforms.ToTensor(),
                ]
            )

    def setup(self, stage: Optional[str] = None):
        self.data_train_source = datasets.ImageFolder(
            self.cfg.full_path + "train_source", transform=self.transform
        )
        self.data_train_target = datasets.ImageFolder(
            self.cfg.full_path + "train_target", transform=self.transform
        )
        self.data_test = datasets.ImageFolder(
            self.cfg.full_path + "test", transform=self.transform
        )

    def train_dataloader(self):
        return [
            DataLoader(
                dataset=self.data_train_source,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                pin_memory=self.cfg.dl.pin_memory,
                shuffle=True,
                drop_last=True,
                multiprocessing_context="fork",
            ),
            DataLoader(
                dataset=self.data_train_target,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                pin_memory=self.cfg.dl.pin_memory,
                shuffle=True,
                drop_last=True,
                multiprocessing_context="fork",
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            dataset=torch.randn(100, 1),
            batch_size=100,
            num_workers=self.cfg.dl.num_workers,
            pin_memory=False,
            multiprocessing_context="fork",
        )

    def get_test_samples(self, batch_size=100):
        test_dl = DataLoader(
            dataset=self.data_test,
            batch_size=batch_size,
            num_workers=self.cfg.dl.num_workers,
            pin_memory=False,
            multiprocessing_context="fork",
            shuffle=True,
        )
        return get_batch(test_dl)
