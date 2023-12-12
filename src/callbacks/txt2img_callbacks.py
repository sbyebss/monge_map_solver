import os
from itertools import zip_longest

import pytorch_lightning as pl  # pylint: disable=unused-import
import torch
from clip import tokenize
from dalle2_pytorch.dalle2_pytorch import l2norm, resize_image_to
from pytorch_lightning import Callback
from torchvision.utils import make_grid

import wandb
from src.callbacks.dalle_laion.loader import load_decoder_from_config

# pylint: disable=arguments-differ,too-many-instance-attributes,line-too-long,too-many-locals


def generate_samples(
    pl_module,
    decoder,
    dalle2_prior,
    example_data,
    device=None,
    skip_ema=False,
):
    """
    Takes example data and generates images from the embeddings
    Returns three lists: real images, generated images, and captions
    """
    # txt -> text imbedding -> OT map -> img embedding -> decoders
    # example_data = [img, emb, txt]
    # img: [B, 3, 224, 224],
    # emb: {"text": [B, 786], "img": [B, 786]}
    # txt: list[str]
    real_images, gt_img_embeddings, text_embeddings, txts = zip(*example_data)

    # We condition on text encodings
    tokenized_texts = tokenize(txts, truncate=True).to(device)
    clip_txt_embed, text_encodings = pl_module.clip.embed_text(tokenized_texts)
    text_encodings = text_encodings.to(device)
    text_embeddings = torch.stack(text_embeddings).to(device)
    gt_img_embeddings = torch.stack(gt_img_embeddings).to(device)
    assert (
        clip_txt_embed - text_embeddings
    ).abs().max() < 5e-3, "precomputed text embeddings and computed on the fly mismatch"

    # ------ Generate images with image embeddings from pretrained prior ------
    prior_params = {}
    prior_params["text_encodings"] = text_encodings
    prior_img_embeddings = dalle2_prior.sample(tokenized_texts, cond_scale=1.0)
    prior_params["image_embed"] = prior_img_embeddings
    samples = decoder.sample(**prior_params)
    prior_emb_img = list(samples)

    # ----- Generate images with ground truth image embeddings----
    gt_img_emb_params = {}
    gt_img_emb_params["text_encodings"] = text_encodings
    gt_img_emb_params["image_embed"] = gt_img_embeddings
    samples = decoder.sample(**gt_img_emb_params)
    # samples=torch.zeros_like(torch.stack(prior_emb_img).to(device))
    img_emb_img = list(samples)

    # ------ Generate images with our method ------
    sample_params = {}
    sample_params["text_encodings"] = text_encodings
    src_text_cond = {"text_embed": text_embeddings, "text_encodings": text_encodings}
    if pl_module.cfg.ema and not skip_ema:
        with pl_module.ema_map.average_parameters():
            img_embeddings = pl_module.map_t(**src_text_cond)
    else:
        img_embeddings = pl_module.map_t(**src_text_cond)

    # we need to normalize back.
    img_embeddings = l2norm(img_embeddings)
    sample_params["image_embed"] = img_embeddings
    samples = decoder.sample(**sample_params)
    # samples=torch.zeros_like(torch.stack(prior_emb_img).to(device))
    generated_img = list(samples)

    # ------ Generate images with text image embedding ------
    caption_only_params = {}
    caption_only_params["text_encodings"] = text_encodings
    caption_only_params["image_embed"] = text_embeddings
    samples = decoder.sample(**caption_only_params)
    # samples=torch.zeros_like(torch.stack(prior_emb_img).to(device))
    caption_only_img = list(samples)

    generated_image_size = generated_img[0].shape[-1]
    real_images = [
        resize_image_to(image, generated_image_size, clamp_range=(0, 1))
        for image in real_images
    ]
    return (
        real_images,
        caption_only_img,
        generated_img,
        prior_emb_img,
        img_emb_img,
        txts,
    )


def generate_grid_samples(
    pl_module, decoder, dalle2_prior, examples, device=None, skip_ema=False
):
    """
    Generates samples and uses torchvision to put them in a side by side grid for easy viewing
    """
    decoder.to(device)
    dalle2_prior.to(device)
    (
        real_images,
        caption_only_imgs,
        generated_imgs,
        prior_emb_imgs,
        img_emb_imgs,
        captions,
    ) = generate_samples(
        pl_module, decoder, dalle2_prior, examples, device=device, skip_ema=skip_ema
    )
    grid_images = [
        make_grid(
            [
                caption_only_img,
                generated_image,
                prior_emb_img,
                img_emb_img,
                original_image,
            ]
        )
        for caption_only_img, generated_image, prior_emb_img, img_emb_img, original_image in zip(
            caption_only_imgs, generated_imgs, prior_emb_imgs, img_emb_imgs, real_images
        )
    ]
    return grid_images, captions


class Text2ImgViz(Callback):
    def __init__(
        self,
        log_interval,
        n_sample_images,
        model_config_path,
        dalle2_decoder_path,
        dalle2_prior_path,
        emb_map_path,
    ) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.n_sample_images = n_sample_images
        self.img_section = "Test Samples"
        # publish_todo: use load_from_config
        if not os.path.exists(dalle2_decoder_path) or not os.path.exists(
            dalle2_prior_path
        ):
            self.decoder, self.prior = load_decoder_from_config(model_config_path)
            torch.save(self.decoder, dalle2_decoder_path)
            torch.save(self.prior, dalle2_prior_path)
        else:
            self.decoder = torch.load(dalle2_decoder_path, map_location="cuda:1")
            self.prior = torch.load(dalle2_prior_path, map_location="cuda:1")
        self.emb_map_path = emb_map_path

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.current_epoch % self.log_interval == 0:
            get_paired_txt_img = trainer.datamodule.get_paired_txt_img
            test_example_data = get_paired_txt_img(
                device=pl_module.device, n=self.n_sample_images
            )
            test_images, test_captions = generate_grid_samples(
                pl_module,
                self.decoder,
                self.prior,
                test_example_data,
                device=pl_module.device,
            )

            wandb_images = [
                wandb.Image(image, caption=caption)
                for image, caption in zip_longest(test_images, test_captions)
            ]
            # Tried both in https://pytorch-lightning.readthedocs.io/en/1.6.0/common/loggers.html#weights-and-biases. They didn't work
            # pl_module.logger.experiment.log({self.img_section: wandb_images})
            # pl_module.logger.log_image(key=self.img_section, images=wandb_images)
            if wandb.run is not None:
                wandb.log({self.img_section: wandb_images})

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.map_t.load_state_dict(torch.load(self.emb_map_path))
