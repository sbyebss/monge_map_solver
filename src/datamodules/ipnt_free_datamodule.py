from typing import Optional

from jamtorch.data import get_batch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.trainimage_dataset import TrainImageDataset
from src.datamodules.datasets.valimage_dataset import ValImageDataset


def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return x


# pylint: disable=W0223


class InpaintImageMaskModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately
        when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called
        before trainer.fit()` or `trainer.test()`."""
        self.data_train = TrainImageDataset(self.cfg)
        self.data_test = ValImageDataset(self.cfg)

    def train_dataloader(self):
        return [
            DataLoader(
                dataset=self.data_train,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_worker,
                shuffle=True,
                drop_last=True,
                multiprocessing_context="fork",
            ),
            DataLoader(
                dataset=self.data_train,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                shuffle=True,
                drop_last=True,
                multiprocessing_context="fork",
            ),
        ]

    def get_test_samples(self, batch_size=100):
        test_dl = DataLoader(
            dataset=self.data_test,
            batch_size=batch_size,
            num_workers=self.cfg.dl.num_workers,
            pin_memory=False,
            multiprocessing_context="fork",
        )
        return get_batch(test_dl)
