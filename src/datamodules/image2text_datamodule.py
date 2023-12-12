from typing import Optional

import numpy as np
from clip import tokenize
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.small_scale_image_dataset import get_img_dataset

# pylint: disable=abstract-method,unspecified-encoding


class ImageTextModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        # We don't have test dataset, we do post-hoc processing
        # directly on the test dataset.
        self.data_source: Optional[Dataset] = None
        self.data_test: Optional[DataLoader] = None
        self.meaningful_class = None
        self.meaningful_token = None
        self.clip_acc = None

    def get_meaningful_token(self):
        prefix = "a photo of a "
        if self.cfg.dataset == "CIFAR100":
            raw_str_label_list = self.data_source.classes

        elif self.cfg.dataset == "imagenet":
            import json

            with open(self.cfg.imagenet_class_json) as f:
                raw_str_label_list = json.load(f)

        self.meaningful_class = np.array([prefix + c for c in raw_str_label_list])
        self.meaningful_token = tokenize(self.meaningful_class)

    def setup(self, stage: Optional[str] = None):
        self.data_source, self.data_test = get_img_dataset(self.cfg)
        # self.data_target, _ = get_img_dataset(self.cfg)
        self.get_meaningful_token()

    def train_dataloader(self):
        src_dl = DataLoader(
            dataset=self.data_source,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            multiprocessing_context="fork",
        )
        return src_dl
        # trg_dl = DataLoader(
        #     dataset=self.data_target,
        #     batch_size=self.cfg.batch_size,
        #     shuffle=True,
        # )
        # return [src_dl, trg_dl]

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            multiprocessing_context="fork",
        )
