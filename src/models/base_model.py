import hydra
import pytorch_lightning as pl
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch_ema import ExponentialMovingAverage

from src.logger.jam_wandb import prefix_metrics_keys
from src.models.loss_zoo import get_general_cost, gradientOptimality, gradientPenalty
from src.viz.img import save_tensor_imgs


def exists(val):
    return val is not None


def degrade_half(x_data):
    x_data[:, :, :, x_data.shape[-1] // 2 :] = 0
    return x_data


# pylint: disable=R0901,abstract-method,line-too-long


class BaseModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cost_func = get_general_cost(cfg.cost_type)

        self.f_net = hydra.utils.instantiate(self.cfg.f_net)
        self.map_t = hydra.utils.instantiate(self.cfg.T_net)

        self.iter_count = 0
        self.ema_map = (
            ExponentialMovingAverage(self.map_t.parameters(), decay=cfg.ema_beta)
            if self.cfg.ema
            else None
        )
        # self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period = cfg.warmup_steps) if exists(cfg.warmup_steps) else None
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        if self.cfg.ema:
            self.ema_map.to(self.device)

    def draw_batch(self, x_data, y_data):
        save_tensor_imgs(
            self.trainer.datamodule.inverse_data_transform(x_data),
            8,
            self.global_step,
            "batch_source",
        )
        save_tensor_imgs(
            self.trainer.datamodule.inverse_data_transform(y_data),
            8,
            self.global_step,
            "batch_target",
        )

    # pylint: disable=arguments-differ,unused-argument

    def training_step(self, batch, batch_idx):
        x_data, y_data = self.get_real_data(batch)
        # pylint: disable=E0633
        optimizer_t, optimizer_f = self.optimizers()

        if self.global_step % (self.cfg.n_outer_iter + self.cfg.n_inner_iter) == 0:
            self.iter_count = 0

        if self.iter_count < self.cfg.n_outer_iter:
            self.opt_f(x_data, y_data, optimizer_f)
        else:
            self.opt_map(x_data, optimizer_t)
        self.iter_count += 1

    def opt_f(self, x_tensor, y_tensor, f_opt, mask=None):
        for param in self.f_net.parameters():
            param.requires_grad = True
        for param in self.map_t.parameters():
            param.requires_grad = False
        loss, loss_info = self.loss_f(x_tensor, y_tensor, mask)
        f_opt.zero_grad()
        self.manual_backward(loss)
        if self.cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.f_net.parameters(), self.cfg.max_grad_norm
            )
        f_opt.step()
        self.log_dict(loss_info)

    def opt_map(self, x_tensor, map_opt, mask=None):
        for param in self.map_t.parameters():
            param.requires_grad = True
        for param in self.f_net.parameters():
            param.requires_grad = False
        loss, loss_info = self.loss_map(x_tensor, mask)
        map_opt.zero_grad()
        self.manual_backward(loss)
        if self.cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.map_t.parameters(), self.cfg.max_grad_norm
            )
        map_opt.step()
        if self.cfg.ema and self.global_step > self.cfg.ema_update_after_step:
            self.ema_map.update()
        self.log_dict(loss_info)

    def loss_f(self, x_tensor, y_tensor, mask=None):
        assert mask is None
        with torch.no_grad():
            tx_tensor = self.map_t(x_tensor)
        f_tx, f_y = self.f_net(tx_tensor).mean(), self.f_net(y_tensor).mean()
        if self.cfg.optimal_penalty:
            gradient_penalty = gradientOptimality(
                self.f_net, tx_tensor, x_tensor, self.cfg.coeff_go
            )
        else:
            gradient_penalty = gradientPenalty(
                self.f_net, y_tensor, tx_tensor, self.cfg.coeff_gp
            )
        f_loss = f_tx - f_y + gradient_penalty
        log_info = prefix_metrics_keys(
            {
                "f_tx": f_tx,
                "f_y": f_y,
                "f_tx - f_y": f_tx - f_y,
                "gradient_penalty": gradient_penalty,
            },
            "f_loss",
        )
        return f_loss, log_info

    def loss_map(self, x_tensor, mask=None):
        assert mask is None
        tx_tensor = self.map_t(x_tensor)
        cost_loss = self.cost_func(x_tensor, tx_tensor, self.cfg.coeff_mse)
        f_tx = self.f_net(tx_tensor).mean()
        map_loss = cost_loss - f_tx
        log_info = prefix_metrics_keys(
            {"cost_loss": cost_loss, "f_tx": f_tx}, "map_loss"
        )
        return map_loss, log_info

    def configure_optimizers(self):
        optimizer_map = optim.Adam(
            self.map_t.parameters(),
            lr=self.cfg.lr_T,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        optimizer_f = optim.Adam(
            self.f_net.parameters(),
            lr=self.cfg.lr_f,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        if self.cfg.schedule_learning_rate:
            return [optimizer_map, optimizer_f], [
                StepLR(
                    optimizer_map,
                    step_size=self.cfg.lr_schedule_epoch,
                    gamma=self.cfg.lr_schedule_scale_t,
                ),
                StepLR(
                    optimizer_f,
                    step_size=self.cfg.lr_schedule_epoch,
                    gamma=self.cfg.lr_schedule_scale_f,
                ),
            ]
        else:
            return optimizer_map, optimizer_f

    def test_step(self, *args, **kwargs):
        pass
