import itertools
from typing import Optional

import torch
from jamtorch.data import get_batch
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.small_scale_image_dataset import get_img_dataset
from src.datamodules.ipnt_fix_datamodule import ImageModule

# This one now only works for CelebA dataset

# pylint: disable=W0223


def target_transform(attr_output, attr_indices, dict_lst):
    attr_indices = torch.tensor(attr_indices)
    attr_vales = attr_output[0, attr_indices]
    new_int_label = dict_lst[str(attr_vales.tolist())]
    return new_int_label


def get_balanced_indices(
    full_ds_attr,
    attr_idx_list,
    num_index_per_class=5000,
    male_ds_flag=1,
    male_attr_idx=20,
):
    # full_ds_attr: torch dataset
    # attr_idx_list: [15, 29] the corresponding attributes in the CelebA
    # num_index_per_class: 5000
    # male_ds_flag: 1 if we want to extract a dataset with all male faces, o.w. 0
    # male_attr_idx: the corresponding attribute in the CelebA
    num_cond = len(attr_idx_list)
    combinations = list(
        itertools.product([0, 1], repeat=num_cond)
    )  # [(0, 0), (0, 1), (1, 0), (1, 1)]
    total_indices = torch.tensor([])
    for condition in combinations:
        subset_cond = full_ds_attr[:, male_attr_idx] == male_ds_flag
        for idx in range(num_cond):
            subset_cond = subset_cond & (
                full_ds_attr[:, attr_idx_list[idx]] == condition[idx]
            )

        subset_indices = subset_cond.nonzero()
        perm = torch.randperm(subset_indices.size(0))
        selected_sub_indices = perm[:num_index_per_class]
        total_indices = torch.cat([total_indices, subset_indices[selected_sub_indices]])
    return total_indices


class ClassImageModule(ImageModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds_full_label: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

        num_attr_cond = len(self.cfg.condition_attr)
        lst_comb = list(itertools.product([0, 1], repeat=num_attr_cond))
        dict_lst = {str(list(lst_comb[i])): i for i in range(len(lst_comb))}
        self.target_transform = lambda x: target_transform(
            x, self.cfg.condition_attr, dict_lst
        )
        self.num_comb = 2**num_attr_cond

    def setup(self, stage: Optional[str] = None):
        self.ds_full_label, self.ds_test = get_img_dataset(
            self.cfg, target_transform=self.target_transform
        )

    def get_balanced_indices(self, num_index_per_class, male=True):
        male_ds_flag = 1 if male else 0
        return get_balanced_indices(
            full_ds_attr=self.ds_full_label.attr,
            attr_idx_list=self.cfg.condition_attr,
            num_index_per_class=num_index_per_class,
            male_ds_flag=male_ds_flag,
            male_attr_idx=self.cfg.male_attr,
        )

    def train_dataloader(self):
        index_female = self.get_balanced_indices(
            male=False, num_index_per_class=self.cfg.num_data_per_class
        )
        index_male = self.get_balanced_indices(
            male=True, num_index_per_class=self.cfg.num_data_per_class
        )
        perm = torch.randperm(index_male.size(0))
        index_female = index_female[perm]
        index_male = index_male[perm]

        female_ds = torch.utils.data.Subset(self.ds_full_label, index_female.long())
        male_ds = torch.utils.data.Subset(self.ds_full_label, index_male.long())
        if self.cfg.direction == "male2female":
            source_ds = male_ds
            target_ds = female_ds
        else:
            source_ds = female_ds
            target_ds = male_ds
        return [
            DataLoader(
                dataset=source_ds,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                shuffle=False,
                drop_last=True,
                multiprocessing_context="fork",
            ),
            DataLoader(
                dataset=target_ds,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                shuffle=False,
                drop_last=True,
                multiprocessing_context="fork",
            ),
        ]

    def test_dataloader(self):
        source_ds_flag = 1 if self.cfg.direction == "male2female" else 0
        test_indices = (
            self.ds_test.attr[:, self.cfg.male_attr] == source_ds_flag
        ).nonzero()
        full_test_ds = torch.utils.data.Subset(self.ds_test, test_indices.long())
        test_dl = DataLoader(
            dataset=full_test_ds,
            batch_size=self.cfg.dl.batch_size,
            num_workers=self.cfg.dl.num_workers,
            multiprocessing_context="fork",
            shuffle=True,
        )

        # ground truth dataset
        gt_ds_flag = 0 if self.cfg.direction == "male2female" else 1
        gt_indices = (self.ds_test.attr[:, self.cfg.male_attr] == gt_ds_flag).nonzero()
        gt_ds = torch.utils.data.Subset(self.ds_test, gt_indices.long())
        gt_dl = DataLoader(
            dataset=gt_ds,
            batch_size=self.cfg.dl.batch_size,
            num_workers=self.cfg.dl.num_workers,
            multiprocessing_context="fork",
            shuffle=True,
        )
        return [test_dl, gt_dl]

    def get_test_samples(self, num_samples=32):
        male_ds_flag = 1 if self.cfg.direction == "male2female" else 0
        test_indices = get_balanced_indices(
            full_ds_attr=self.ds_test.attr,
            attr_idx_list=self.cfg.condition_attr,
            num_index_per_class=int(num_samples / self.num_comb),
            male_ds_flag=male_ds_flag,
            male_attr_idx=self.cfg.male_attr,
        )

        balanced_test_ds = torch.utils.data.Subset(self.ds_test, test_indices.long())
        test_dl = DataLoader(
            dataset=balanced_test_ds,
            batch_size=(len(balanced_test_ds)),
            num_workers=self.cfg.dl.num_workers,
            multiprocessing_context="fork",
            shuffle=False,
        )
        return get_batch(test_dl)
