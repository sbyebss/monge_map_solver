import random

import numpy as np
import torch
from skimage.util.shape import view_as_windows

from src.logger.jam_wandb import prefix_metrics_keys
from src.models.inpaint_fix_mask import InpaintFixMaskModule
from src.models.loss_zoo import gradientOptimality, gradientPenalty


def rectangle_mask_batched(
    image_height=64, image_width=64, min_hole_size=32, max_hole_size=32, batch_size=100
):
    hole_size = random.randint(min_hole_size, max_hole_size)
    hole_size = min(int(image_width * 0.8), int(image_height * 0.8), hole_size)

    w_off = image_width - hole_size + 1
    h_off = image_height - hole_size + 1
    mask_area = w_off * h_off
    # pylint: disable=unbalanced-tuple-unpacking
    h_idxes, w_idxes = np.unravel_index(
        np.random.choice(mask_area, size=batch_size, replace=True), (h_off, w_off)
    )

    mask_out = np.zeros(shape=(batch_size, image_height, image_width), dtype=bool)
    mask = view_as_windows(mask_out, (1, hole_size, hole_size))[..., 0, :, :]
    mask[np.arange(len(h_idxes)), h_idxes, w_idxes] = 1
    mask_out = torch.from_numpy(mask_out).float()
    mask_out = mask_out[:, None, ...]
    return mask_out


def rectangle_mask_uniform(
    image_height=64, image_width=64, min_hole_size=32, max_hole_size=32, batch_size=100
):
    mask = torch.zeros((image_height, image_width))
    hole_size = random.randint(min_hole_size, max_hole_size)
    hole_size = min(int(image_width * 0.8), int(image_height * 0.8), hole_size)
    x = random.randint(0, image_width - hole_size - 1)
    y = random.randint(0, image_height - hole_size - 1)
    mask[x : x + hole_size, y : y + hole_size] = 1
    mask = mask[None, None, ...].expand(batch_size, -1, -1, -1)
    return mask.float()


def degrade_rectangle(x_data, mask_type="batch_vary"):
    batch_size, _, height, weigh = x_data.shape
    # the mask is 1 at the degraded area, 0 elsewhere
    if mask_type == "uniform":
        mask = rectangle_mask_uniform(
            height, weigh, min(height, weigh) // 2, min(height, weigh) // 2, batch_size
        )
    elif mask_type == "batch_vary":
        mask = rectangle_mask_batched(
            height, weigh, min(height, weigh) // 2, min(height, weigh) // 2, batch_size
        )
    mask = mask.to(x_data.device)
    inputs = x_data * (1 - mask)
    return inputs, mask


# pylint: disable=abstract-method,too-many-ancestors


class InpaintFreeMaskModule(InpaintFixMaskModule):
    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):
        x_data, mask, y_data = self.get_real_data(batch)
        # pylint: disable=E0633
        if self.global_step % (self.cfg.n_outer_iter + self.cfg.n_inner_iter) == 0:
            self.iter_count = 0
        optimizer_t, optimizer_f = self.optimizers()

        if self.iter_count < self.cfg.n_outer_iter:
            self.opt_f(x_data, y_data, optimizer_f, mask)
        else:
            self.opt_map(x_data, optimizer_t, mask)

        self.iter_count += 1

    def get_real_data(self, batch):
        d_fn = self.trainer.datamodule.data_transform
        x_data, y_data = batch
        x_data, y_data = d_fn(x_data[0]), d_fn(y_data[0])
        inputs, mask = degrade_rectangle(x_data, mask_type=self.cfg.mask_type)
        if self.global_step == 1:
            self.draw_batch(inputs, y_data)
        return inputs, mask, y_data

    def loss_f(self, x_tensor, y_tensor, mask=None):
        tx_tensor = self.map_t(x_tensor)
        composite_img = x_tensor * (1 - mask) + tx_tensor * mask
        f_tx, f_y = self.f_net(composite_img).mean(), self.f_net(y_tensor).mean()
        if self.cfg.optimal_penalty:
            gradient_penalty = gradientOptimality(
                self.f_net, composite_img, x_tensor, self.cfg.coeff_go
            )
        else:
            gradient_penalty = gradientPenalty(
                self.f_net, y_tensor, composite_img, self.cfg.coeff_gp
            )
        f_loss = f_tx - f_y + gradient_penalty
        log_info = prefix_metrics_keys(
            {"f_tx": f_tx, "f_y": f_y, "gradient_penalty": gradient_penalty},
            "f_loss",
        )
        return f_loss, log_info

    def loss_map(self, x_tensor, mask=None):
        assert mask is not None
        tx_tensor = self.map_t(x_tensor)
        composite_img = x_tensor * (1 - mask) + tx_tensor * mask
        if self.cfg.masked_cost:
            cost_loss = self.cost_func(
                x_tensor * (1 - mask), tx_tensor * (1 - mask), self.cfg.coeff_mse
            )
        else:
            cost_loss = self.cost_func(x_tensor, tx_tensor, self.cfg.coeff_mse)
        f_tx = self.f_net(composite_img).mean()
        map_loss = cost_loss - f_tx
        log_info = prefix_metrics_keys(
            {"cost_loss": cost_loss, "f_tx": f_tx}, "map_loss"
        )
        return map_loss, log_info
