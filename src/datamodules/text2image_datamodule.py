from functools import partial
from typing import Optional

import omegaconf
import webdataset as wds
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader, get_reader
from dalle2_pytorch.dataloaders.prior_loader import PriorEmbeddingDataset
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.datamodules.utils import get_example_data


# pylint: disable=abstract-method,inconsistent-return-statements,too-many-instance-attributes
def get_transformation(preprocessing):
    def _get_transformation(transformation_name, **kwargs):
        if transformation_name == "RandomResizedCrop":
            return T.RandomResizedCrop(**kwargs)
        if transformation_name == "RandomHorizontalFlip":
            return T.RandomHorizontalFlip()
        if transformation_name == "ToTensor":
            return T.ToTensor()

    transforms = []
    for transform_name, transform_kwargs_or_bool in preprocessing.items():
        transform_kwargs = (
            {}
            if not isinstance(transform_kwargs_or_bool, omegaconf.DictConfig)
            else transform_kwargs_or_bool
        )
        transforms.append(_get_transformation(transform_name, **transform_kwargs))
    img_process = T.Compose(transforms)

    return img_process


class EmbeddingDataset(PriorEmbeddingDataset):
    def __len__(self):
        return (self.stop - self.start) // self.batch_size + 1


class TextImageEmbedModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        self.cfg.num_train_data = int(cfg.num_train_data)
        self.cfg.num_test_data = int(cfg.num_test_data)

        self.img_reader = get_reader(
            text_conditioned=True,
            img_url=cfg.img_url,
            meta_url=cfg.meta_url,
        )
        self.data_train_source: Optional[Dataset] = None
        self.data_train_target: Optional[Dataset] = None
        self.data_test_sml: Optional[Dataset] = None
        self.data_test_sampling_dl: Optional[DataLoader] = None

    def setup(self, stage: Optional[str] = None):
        # Use outright different data for source and target
        # Now the train / test dataloaders above are from discordered
        # clip-retrieval embeddings, it's img_emb.npy files from up to down.
        self.data_train_source = EmbeddingDataset(
            text_conditioned=True,
            batch_size=self.cfg.batch_size,
            start=0,
            stop=self.cfg.num_train_data,
            image_reader=self.img_reader,
        )

        self.data_train_target = EmbeddingDataset(
            text_conditioned=True,
            batch_size=self.cfg.batch_size,
            start=self.cfg.num_train_data,
            stop=2 * self.cfg.num_train_data,
            # start=0,
            # stop=self.cfg.num_train_data,
            image_reader=self.img_reader,
        )

        # ---------- This is used for cosine similarity ---------
        self.data_test_sml = EmbeddingDataset(
            text_conditioned=True,
            batch_size=self.cfg.batch_size,
            start=2 * self.cfg.num_train_data,
            stop=2 * self.cfg.num_train_data + self.cfg.num_test_data,
            image_reader=self.img_reader,
        )

        # ---------- This is used for quality evaluation ---------
        img_process = get_transformation(self.cfg.preprocessing)
        available_shards = list(range(self.cfg.start_shard, self.cfg.end_shard + 1))
        tar_urls = [
            self.cfg.webdataset_base_url.format(str(shard).zfill(self.cfg.shard_width))
            for shard in available_shards
        ]
        # TODO: how to tell whether this dl is in train or test?
        self.data_test_sampling_dl = create_image_embedding_dataloader(
            tar_url=tar_urls,
            num_workers=12,
            batch_size=self.cfg.sample_batch_size,
            img_embeddings_url=self.cfg.img_embeddings_url,
            text_embeddings_url=self.cfg.text_embeddings_url,
            index_width=self.cfg.index_width,
            extra_keys=["txt"],
            shuffle_num=1,
            shuffle_shards=True,
            resample_shards=False,
            img_preproc=img_process,
            handler=wds.handlers.warn_and_continue,
        )

    def train_dataloader(self):
        src_dl = DataLoader(
            dataset=self.data_train_source,
            batch_size=None,
        )
        trg_dl = DataLoader(
            dataset=self.data_train_target,
            batch_size=None,
        )
        return [src_dl, trg_dl]

    def val_dataloader(self):
        data_test_sml_dl = DataLoader(
            dataset=self.data_test_sml,
            batch_size=None,
        )
        return data_test_sml_dl

    def test_dataloader(self):
        data_test_sml_dl = DataLoader(
            dataset=self.data_test_sml,
            batch_size=None,
        )
        return [data_test_sml_dl, self.data_test_sampling_dl]
        # return [data_test_sml_dl]

    @property
    def get_paired_txt_img(self):
        return partial(get_example_data, dataloader=self.data_test_sampling_dl)
