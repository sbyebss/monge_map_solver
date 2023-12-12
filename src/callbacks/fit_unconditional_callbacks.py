import torch

from src.callbacks.img_callbacks import MapViz
from src.viz.img import save_tensor_imgs


class FitMapViz(MapViz):
    def pushforward_images(self, trainer, pl_module, name_idx, skip_ema=False):
        with torch.no_grad():
            source_feat, source_label = trainer.datamodule.get_test_samples(
                self.num_test_sample
            )
            source_feat = source_feat.to(pl_module.device)
            source_label = source_label.to(pl_module.device)
            source_feat = trainer.datamodule.data_transform(source_feat)

            canonical_feat = pl_module.canonical_map(source_feat, source_label)
            # In the training, we need ema manually,
            # but in the test, the model is already after averaged.
            if skip_ema or not pl_module.cfg.ema:
                output_feat = pl_module.map_t(source_feat)
            else:
                with pl_module.ema_map.average_parameters():
                    output_feat = pl_module.map_t(source_feat)

        output_combo = torch.cat([canonical_feat, output_feat], dim=0)
        save_tensor_imgs(
            trainer.datamodule.inverse_data_transform(output_combo),
            self.num_sample_per_row,
            name_idx,
            "pushforward",
        )
