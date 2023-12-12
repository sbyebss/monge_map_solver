from src.models.base_model import BaseModule


def degrade_half(x_data):
    x_data[:, :, :, x_data.shape[-1] // 2 :] = 0
    return x_data


# pylint: disable=R0901,abstract-method


class InpaintFixMaskModule(BaseModule):
    def get_real_data(self, batch):
        d_fn = self.trainer.datamodule.data_transform
        x_data, y_data = batch
        x_data, y_data = d_fn(x_data[0]), d_fn(y_data[0])
        x_data = degrade_half(x_data)
        if self.global_step == 1:
            self.draw_batch(x_data, y_data)
        return x_data, y_data
