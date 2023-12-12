from os import makedirs, path

import hydra
import torch
import torch.nn.functional as F
import torch_fidelity
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification.accuracy import Accuracy

from src.logger.jam_wandb import prefix_metrics_keys
from src.models.base_model import BaseModule
from src.models.loss_zoo import label_cost
from src.utils import lht_utils
from src.viz.img import save_seperate_imgs

log = lht_utils.get_logger(__name__)

ce_loss = nn.CrossEntropyLoss()
# pylint: disable=abstract-method,too-many-ancestors,arguments-renamed,line-too-long,arguments-differ,unused-argument,too-many-locals,bare-except


def turn_off_grad(network):
    for param in network.parameters():
        if not param.requires_grad:
            break
        param.requires_grad = False


def get_feat_label(feat_label):
    if isinstance(feat_label, list):  # image
        return feat_label
    else:
        return None


class ClassMapModule(BaseModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.classifier = hydra.utils.instantiate(self.cfg.classifier)
        self.load_classifier()
        self.w_distance_table = torch.ones([cfg.num_class, cfg.num_class])
        self.w_distance_table.fill_diagonal_(0)
        self.train_acc = Accuracy()
        self.validate_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        source_data, target_data = self.get_real_data(batch)
        # pylint: disable=E0633
        opt_t, opt_f, opt_l = self.optimizers()
        if self.current_epoch < self.cfg.classifier_epoch:
            self.pretrain_feature_map(source_data, opt_t)
            if self.pretrain_clsf and self.cfg.coeff_label > 0:
                self.pretrain_classifier(target_data, opt_l)
            else:
                self.test_classifier(target_data)
        else:
            turn_off_grad(self.classifier)
            self.opt_f_g(source_data, target_data, opt_f, opt_t)

    def pretrain_feature_map(self, data, map_opt):
        feat, source_label = get_feat_label(data)
        loss = F.mse_loss(self.map_t(feat, source_label), feat)
        if loss > 1e-3:
            map_opt.zero_grad()
            loss.backward()
            map_opt.step()
            self.log_dict(
                prefix_metrics_keys(
                    {"id_loss": loss},
                    "pretrain_loss",
                )
            )

    def opt_f_g(self, source_data, target_data, f_opt, map_opt):
        if self.global_step % (self.cfg.n_outer_iter + self.cfg.n_inner_iter) == 0:
            self.iter_count = 0

        if self.iter_count < self.cfg.n_outer_iter:
            self.opt_f(source_data, target_data, f_opt)
        else:
            self.opt_map(source_data, map_opt)
        self.iter_count += 1

    def test_classifier(self, target_data):
        _, loss_info = self.loss_classify(target_data)
        self.log_dict(loss_info)

    def pretrain_classifier(self, target_data, optimizer_l):
        loss, loss_info = self.loss_classify(target_data)
        if loss > 1e-3:
            optimizer_l.zero_grad()
            self.manual_backward(loss)
            optimizer_l.step()
            self.log_dict(loss_info)

    def get_real_data(self, batch):
        d_fn = self.trainer.datamodule.data_transform
        x_data, y_data = batch
        x_data[0], y_data[0] = d_fn(x_data[0]), d_fn(y_data[0])
        if self.global_step == 0:
            self.draw_batch(x_data[0], y_data[0])
        return x_data, y_data

    def load_classifier(self):
        if path.exists(self.cfg.classifier_save_path):
            try:
                self.classifier.load_state_dict(
                    torch.load(self.cfg.classifier_save_path)
                )
            except:
                self.classifier.load_state_dict(
                    torch.load(self.cfg.classifier_save_path)["model_state_dict"]
                )
            log.info(
                f"Successfully load the pretrained classifier from <{self.cfg.classifier_save_path }>"
            )
            self.pretrain_clsf = False
            turn_off_grad(self.classifier)
        else:
            log.info(
                "Didn't find the pretrained classifier, need to train it from scratch..."
            )
            self.pretrain_clsf = True

    def loss_classify(self, feat_label):
        target_feat, target_label = get_feat_label(feat_label)
        target_label = target_label.long()
        # batch_size = target_feat.shape[0]
        # this prob output is unnormalized.
        label_logits = self.classifier(target_feat)
        loss = ce_loss(label_logits, target_label)
        pred = torch.argmax(label_logits, dim=1)
        log_info = prefix_metrics_keys(
            {"ce_loss": loss, "accuracy": self.train_acc(pred, target_label)},
            "pretrain_loss",
        )
        return loss, log_info

    def loss_f(self, source_feat_label, target_feat_label, mask=None):
        source_feat, source_label = get_feat_label(source_feat_label)
        with torch.no_grad():
            output_feat = self.map_t(source_feat, source_label)
            self.classifier.eval()
            label_logits = self.classifier(output_feat)
            label_probs = F.softmax(label_logits, dim=1)
        target_feat, target_label = get_feat_label(target_feat_label)

        source_label = source_label.view(-1).long()
        # source_label_onehot = F.one_hot(
        #     source_label, num_classes=self.cfg.num_class
        # ).float()

        target_label = target_label.view(-1).long()
        target_label_onehot = F.one_hot(
            target_label, num_classes=self.cfg.num_class
        ).float()
        # Use pushforward label. Souce label makes it unstable.
        f_tx, f_y = (
            self.f_net(output_feat, label_probs).mean(),
            self.f_net(target_feat, target_label_onehot).mean(),
        )
        f_loss = f_tx - f_y
        log_info = prefix_metrics_keys(
            {"f_tx": f_tx, "f_y": f_y, "f_tx - f_y": f_tx - f_y},
            "f_loss",
        )
        return f_loss, log_info

    def loss_map(self, source_feat_label, mask=None):
        source_feat, source_label = get_feat_label(source_feat_label)
        output_feat = self.map_t(source_feat, source_label)
        self.classifier.eval()
        label_logits = self.classifier(output_feat)
        label_probs = F.softmax(label_logits, dim=1)

        feat_loss = self.cost_func(source_feat, output_feat, self.cfg.coeff_mse)
        self.w_distance_table = self.w_distance_table.to(self.device)
        label_loss = label_cost(
            self.w_distance_table, source_label, label_probs, self.cfg.coeff_label
        )
        cost_loss = feat_loss + label_loss

        f_tx = self.f_net(output_feat, label_probs).mean()
        map_loss = cost_loss - f_tx
        log_info = prefix_metrics_keys(
            {
                "cost_loss": cost_loss,
                "feat_loss": feat_loss,
                "label_loss": label_loss,
                "f_tx": f_tx,
            },
            "map_loss",
        )
        return map_loss, log_info

    def on_test_start(self) -> None:
        self.map_t.load_state_dict(torch.load(self.cfg.map_ckpt_path))

    def validation_step(self, batch, batch_idx, dataloader_id):
        if not path.exists(self.cfg.fid_fake_img_path):
            makedirs(self.cfg.fid_fake_img_path)
        if not path.exists(self.cfg.real_img_path):
            makedirs(self.cfg.real_img_path)
        cnt = batch_idx * self.trainer.datamodule.cfg.dl.batch_size
        if dataloader_id == 0:
            d_fn = self.trainer.datamodule.data_transform
            source_feat, source_label = get_feat_label(batch)
            source_feat = d_fn(source_feat)
            if self.cfg.ema:
                with self.ema_map.average_parameters():
                    generated_img = self.map_t(source_feat, source_label)
            save_seperate_imgs(
                self.trainer.datamodule.inverse_data_transform(generated_img),
                self.cfg.fid_fake_img_path,
                cnt,
            )
            label_logits = self.classifier(generated_img)
            pred = torch.argmax(label_logits, dim=1)
            self.validate_acc.update(pred, source_label)
        else:
            target_feat, _ = get_feat_label(batch)
            save_seperate_imgs(target_feat, self.cfg.real_img_path, cnt)

    def validation_epoch_end(self, outputs) -> None:
        # print("Begin to calculate validation accuracy...")
        total_valid_accuracy = self.validate_acc.compute()
        self.validate_acc.reset()
        # print("Begin to calculate FID score...")
        metric = torch_fidelity.calculate_metrics(
            input1=self.cfg.fid_fake_img_path,
            input2=self.cfg.real_img_path,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        self.log_dict(
            {
                "validate/fid": metric["frechet_inception_distance"],
                "validate/acc": total_valid_accuracy,
            }
        )

    def test_step(self, batch, batch_idx, dataloader_id):
        if path.exists(self.cfg.fid_fake_img_path) and path.exists(
            self.cfg.real_img_path
        ):
            return

        makedirs(self.cfg.fid_fake_img_path)
        makedirs(self.cfg.real_img_path)

        cnt = batch_idx * self.trainer.datamodule.cfg.dl.batch_size
        if dataloader_id == 0:
            d_fn = self.trainer.datamodule.data_transform
            source_feat, source_label = get_feat_label(batch)
            source_feat = d_fn(source_feat)
            generated_img = self.map_t(source_feat, source_label)
            save_seperate_imgs(
                self.trainer.datamodule.inverse_data_transform(generated_img),
                self.cfg.fid_fake_img_path,
                cnt,
            )
        else:
            target_feat, _ = get_feat_label(batch)
            save_seperate_imgs(target_feat, self.cfg.real_img_path, cnt)

    def test_epoch_end(self, output) -> None:
        if path.exists(self.cfg.fid_fake_img_path):
            return None

        metric = torch_fidelity.calculate_metrics(
            input1=self.cfg.fid_fake_img_path,
            input2=self.cfg.real_img_path,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        torch.save(metric, "fid_result.pth")
        return metric

    def configure_optimizers(self):
        optimizer_map = optim.Adam(
            self.map_t.parameters(),
            lr=self.cfg.lr_T,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=1e-10,
        )
        optimizer_f = optim.Adam(
            self.f_net.parameters(),
            lr=self.cfg.lr_f,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=1e-10,
        )
        optimizer_l = optim.Adam(
            self.classifier.parameters(), lr=self.cfg.lr_l, weight_decay=1e-10
        )
        if self.cfg.schedule_learning_rate:
            return [optimizer_map, optimizer_f, optimizer_l], [
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
                StepLR(
                    optimizer_l,
                    step_size=self.cfg.lr_schedule_epoch,
                    gamma=self.cfg.lr_schedule_scale_l,
                ),
            ]
        else:
            return optimizer_map, optimizer_f, optimizer_l
