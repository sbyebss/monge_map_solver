import errno
from os import strerror

import hydra
import torch
import torch.nn.functional as F
from torch import optim

from src.models.class_model import ClassMapModule, get_feat_label, turn_off_grad
from src.utils import lht_utils

log = lht_utils.get_logger(__name__)

# pylint: disable=abstract-method,too-many-ancestors,arguments-renamed,line-too-long,arguments-differ,unused-argument,too-many-locals,bare-except


class FitClassMapModule(ClassMapModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        if self.pretrain_clsf:
            raise FileNotFoundError(
                errno.ENOENT, strerror(errno.ENOENT), "pretrained classifier"
            )
        self.canonical_map = hydra.utils.instantiate(self.cfg.canonical_map)
        self.load_canonical_map()

    def load_canonical_map(self):
        self.canonical_map.load_state_dict(torch.load(self.cfg.canonical_map_save_path))

    def training_step(self, batch, batch_idx):
        source_data, _ = self.get_real_data(batch)
        # pylint: disable=E0633
        opt_t = self.optimizers()
        turn_off_grad(self.classifier)
        turn_off_grad(self.canonical_map)
        self.opt_map(source_data, opt_t)

    def get_real_data(self, batch):
        d_fn = self.trainer.datamodule.data_transform
        x_data, y_data = batch
        x_data[0], y_data[0] = d_fn(x_data[0]), d_fn(y_data[0])
        if self.global_step == 0:
            pf_data = self.canonical_map(x_data[0], x_data[1])
            self.draw_batch(x_data[0], pf_data)
        return x_data, y_data

    def loss_map(self, source_feat_label, mask=None):
        source_feat, source_label = get_feat_label(source_feat_label)
        canonical_feat = self.canonical_map(source_feat, source_label)
        mse_loss = F.mse_loss(self.map_t(source_feat), canonical_feat)
        log_info = {"map_loss/identity_loss": mse_loss}
        return mse_loss, log_info

    def configure_optimizers(self):
        optimizer_map = optim.Adam(
            self.map_t.parameters(),
            lr=self.cfg.lr_T,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=1e-10,
        )
        return optimizer_map
