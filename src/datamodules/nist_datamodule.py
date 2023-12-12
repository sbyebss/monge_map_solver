from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from jamtorch.data import get_batch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.small_scale_image_dataset import get_img_dataset
from src.datamodules.ipnt_fix_datamodule import ImageModule

# pylint: disable=W0223


def transform2torch(dataset_input):
    dataset = deepcopy(dataset_input)  # This looks bad!!!
    if isinstance(dataset.data, np.ndarray):
        dataset.targets = torch.Tensor(dataset.targets)
    elif isinstance(dataset.data, list):
        dataset.data = torch.Tensor(dataset.data)
        dataset.targets = torch.Tensor(dataset.targets)
    # Some dataset's label starts from 1 such as EMNIST, so we need to shift it
    if isinstance(dataset.targets, torch.Tensor) and min(dataset.targets) == 1:
        dataset.targets -= 1
    return dataset


def extract_class(dataset_input, index):
    dataset = transform2torch(dataset_input)
    indices = dataset.targets == index
    dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
    return dataset


def get_nist_num_label(dataset_name: str):
    if dataset_name == "EMNIST":
        return 26
    else:
        return 10


class NIST(ImageModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        self.source_train_data: Optional[Dataset] = None
        self.target_train_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.target_gt_data: Optional[Dataset] = None
        self.sorted_test_data = []
        self.source_class = get_nist_num_label(self.cfg.source.dataset)

    def prepare_data(self):
        get_img_dataset(self.cfg.source)
        get_img_dataset(self.cfg.target)

    def setup(self, stage: Optional[str] = None):
        self.source_train_data, self.test_data = get_img_dataset(self.cfg.source)
        self.target_train_data, self.target_gt_data = get_img_dataset(self.cfg.target)
        self.preprocess_label()
        self.sort_test_data()

    def preprocess_label(self):
        if self.cfg.source.dataset == "EMNIST":
            self.source_train_data = transform2torch(self.source_train_data)
            self.test_data = transform2torch(self.test_data)
        if self.cfg.target.dataset == "EMNIST":
            self.target_train_data = transform2torch(self.target_train_data)
            self.target_gt_data = transform2torch(self.target_gt_data)

    def sort_test_data(self):
        for cls_idx in range(self.source_class):
            self.sorted_test_data.append(extract_class(self.test_data, cls_idx))

    def train_dataloader(self):
        return [
            DataLoader(
                dataset=self.source_train_data,
                batch_size=self.cfg.dl.batch_size,
                # num_workers=self.cfg.dl.num_workers,
                # pin_memory=self.cfg.dl.pin_memory,
                # multiprocessing_context="fork",
                shuffle=True,
                drop_last=True,
            ),
            DataLoader(
                dataset=self.target_train_data,
                batch_size=self.cfg.dl.batch_size,
                # num_workers=self.cfg.dl.num_workers,
                # pin_memory=self.cfg.dl.pin_memory,
                # multiprocessing_context="fork",
                shuffle=True,
                drop_last=True,
            ),
        ]

    def val_dataloader(self):
        test_dl = DataLoader(
            dataset=self.test_data,
            batch_size=self.cfg.dl.batch_size,
            # num_workers=self.cfg.dl.num_workers,
            # pin_memory=self.cfg.dl.pin_memory,
            # multiprocessing_context=get_context('loky'),
            shuffle=True,
        )
        gt_dl = DataLoader(
            dataset=self.target_gt_data,
            batch_size=self.cfg.dl.batch_size,
            # num_workers=self.cfg.dl.num_workers,
            # pin_memory=self.cfg.dl.pin_memory,
            # multiprocessing_context=get_context('loky'),
            shuffle=True,
        )
        return [test_dl, gt_dl]

    def get_test_samples(self, batch_size=100, shuffle=False):
        num_class = self.source_class
        num_per_class = int(batch_size / num_class)
        test_feat = torch.zeros(
            [
                num_per_class,
                num_class,
                self.cfg.channel,
                self.cfg.source.image_size,
                self.cfg.source.image_size,
            ]
        )
        test_label = torch.zeros([num_per_class, num_class]).to(torch.int64)
        for cls_idx in range(num_class):
            dataloader = DataLoader(
                dataset=self.sorted_test_data[cls_idx],
                batch_size=num_per_class,
                shuffle=shuffle,
                # num_workers=self.cfg.dl.num_workers,
                # pin_memory=self.cfg.dl.pin_memory,
                # multiprocessing_context="fork",
            )
            test_feat[:, cls_idx], test_label[:, cls_idx] = get_batch(dataloader)
        test_label = test_label.view(-1)
        test_feat = rearrange(test_feat, "b n c h w -> (b n) c h w")
        return [test_feat, test_label]
