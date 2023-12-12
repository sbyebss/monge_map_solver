import torch
from pytorch_lightning import Callback


class SaveCb(Callback):
    def __init__(self, save_interval=1, wait_epoch=0, dump_f=True) -> None:
        super().__init__()
        self.save_interval = save_interval
        self.wait_epoch = wait_epoch
        self.dump_f = dump_f

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if (pl_module.current_epoch + 1) > self.wait_epoch and (
            pl_module.current_epoch + 1
        ) % self.save_interval == 0:
            if pl_module.cfg.ema:
                with pl_module.ema_map.average_parameters():
                    torch.save(
                        pl_module.map_t.state_dict(),
                        f"map_{pl_module.current_epoch+1}_ema.pth",
                    )
            else:
                torch.save(
                    pl_module.map_t.state_dict(), f"map_{pl_module.current_epoch+1}.pth"
                )
            if self.dump_f:
                torch.save(
                    pl_module.f_net.state_dict(),
                    f"f_net_{pl_module.current_epoch+1}.pth",
                )


class SaveClassifierCb(Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if (
            pl_module.cfg.dump_classifier
            and pl_module.current_epoch == pl_module.cfg.classifier_epoch - 1
        ):
            torch.save(
                pl_module.classifier.state_dict(), pl_module.cfg.classifier_save_path
            )
