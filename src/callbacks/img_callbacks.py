import pytorch_lightning as pl  # pylint: disable=unused-import
import torch
from pytorch_lightning import Callback

from src.models.inpaint_fix_mask import degrade_half
from src.models.inpaint_free_mask import rectangle_mask_batched
from src.viz.img import save_tensor_imgs


def degrade_rectangle(x_data):
    batch_size, _, height, weigh = x_data.shape
    # the mask is 1 at the degraded area, 0 elsewhere
    mask = rectangle_mask_batched(
        height, weigh, min(height, weigh) // 2, min(height, weigh) // 2, batch_size
    )
    mask = mask.to(x_data.device)
    inputs = x_data * (1 - mask)
    return inputs, mask


def test_push_images(datamodule, pl_module):
    with torch.no_grad():
        source = datamodule.get_test_samples(64)[0].to(pl_module.device)
        real_img = source
        source = datamodule.data_transform(source)

        if type(pl_module).__name__ == "InpaintFixMaskModule":
            source = degrade_half(source)
            mask = None
        else:
            source, mask = degrade_rectangle(source)
        if pl_module.cfg.ema:
            with pl_module.ema_map.average_parameters():
                generated_img = pl_module.map_t(source)
        else:
            generated_img = pl_module.map_t(source)
        composite_img = (
            source * (1 - mask) + generated_img * mask if mask is not None else None
        )
        return source, composite_img, generated_img, real_img


# pylint: disable=arguments-differ,too-many-instance-attributes, line-too-long


class InpaintViz(Callback):
    def __init__(self, log_interval) -> None:
        super().__init__()
        self.log_interval = log_interval

    def on_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.global_step % self.log_interval == 0:
            degraded_img, composite_img, generated_img, real_img = test_push_images(
                trainer.datamodule, pl_module
            )
            save_tensor_imgs(
                trainer.datamodule.inverse_data_transform(degraded_img),
                8,
                pl_module.global_step,
                "degraded_images",
            )
            if composite_img is not None:
                save_tensor_imgs(
                    trainer.datamodule.inverse_data_transform(composite_img),
                    8,
                    pl_module.global_step,
                    "composite_img",
                )
            save_tensor_imgs(
                trainer.datamodule.inverse_data_transform(generated_img),
                8,
                pl_module.global_step,
                "pushed_images",
            )
            save_tensor_imgs(
                real_img,
                8,
                pl_module.global_step,
                "real_img",
            )


class MapViz(Callback):
    def __init__(self, log_interval, num_test_sample, num_sample_per_row=10) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.num_test_sample = num_test_sample
        self.num_sample_per_row = num_sample_per_row

    def on_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.global_step % self.log_interval == 0:
            self.pushforward_images(trainer, pl_module, pl_module.global_step)

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        for idx in range(100):
            self.pushforward_images(trainer, pl_module, idx, skip_ema=True)

    def pushforward_images(self, trainer, pl_module, name_idx, skip_ema=False):
        with torch.no_grad():
            source_feat, source_label = trainer.datamodule.get_test_samples(
                self.num_test_sample
            )
            source_feat = source_feat.to(pl_module.device)
            source_label = source_label.to(pl_module.device)
            source_feat = trainer.datamodule.data_transform(source_feat)
            # In the training, we need ema manually,
            # but in the test, the model is already after averaged.
            if skip_ema or not pl_module.cfg.ema:
                output_feat = pl_module.map_t(source_feat, source_label)
            else:
                with pl_module.ema_map.average_parameters():
                    output_feat = pl_module.map_t(source_feat, source_label)

        source_output = torch.cat([source_feat, output_feat], dim=0)
        save_tensor_imgs(
            trainer.datamodule.inverse_data_transform(source_output),
            self.num_sample_per_row,
            name_idx,
            "pushforward",
        )
