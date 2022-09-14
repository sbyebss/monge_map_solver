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


# pylint: disable=arguments-differ,too-many-instance-attributes


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
