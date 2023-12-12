import torch
import torch.nn.functional as F
from dalle2_pytorch.dalle2_pytorch import CausalTransformer, exists, l2norm, prob_mask_like
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


# pylint: disable=line-too-long,too-many-instance-attributes,too-many-locals, unused-argument
class DiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim=768,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        max_text_len=77,
        **kwargs
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds

        self.to_text_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_text_embeds)
            if num_text_embeds > 1
            else nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_text_embeds),
        )

        self.to_image_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_image_embeds)
            if num_image_embeds > 1
            else nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_image_embeds),
        )

        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim=dim, **kwargs)

        # dalle1 learned padding strategy

        self.max_text_len = max_text_len
        self.null_text_encodings = nn.Parameter(torch.randn(1, max_text_len, dim))
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, dim))

        self.image_embed_scale = dim**0.5

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    def forward(
        self, text_embed=None, text_encodings=None, cond_drop_prob=0.1, **kwargs
    ):
        # We normalize both text_emb and the output pushforward img_emb.
        text_embed = self.l2norm_clamp_embed(text_embed)
        batch, dim, device, dtype = (
            *text_embed.shape,
            text_embed.device,
            text_embed.dtype,
        )

        # in section 2.2, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep embedding, noised CLIP image embedding, final embedding for prediction"

        text_embed = self.to_text_embeds(text_embed)
        text_keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")

        # make text encodings optional
        # although the paper seems to suggest it is present <--

        if not exists(text_encodings):
            text_encodings = torch.empty((batch, 0, dim), device=device, dtype=dtype)

        mask = torch.any(text_encodings != 0.0, dim=-1)

        # replace any padding in the text encodings with learned padding tokens unique across position

        text_encodings = text_encodings[:, : self.max_text_len]
        mask = mask[:, : self.max_text_len]

        text_len = text_encodings.shape[-2]
        remainder = self.max_text_len - text_len

        if remainder > 0:
            text_encodings = F.pad(text_encodings, (0, 0, 0, remainder), value=0.0)
            mask = F.pad(mask, (0, remainder), value=False)

        # mask out text encodings with null encodings

        null_text_encodings = self.null_text_encodings.to(text_encodings.dtype)

        text_encodings = torch.where(
            rearrange(mask, "b n -> b n 1").clone(), text_encodings, null_text_encodings
        )

        # mask out text embeddings with null text embeddings

        null_text_embeds = self.null_text_embed.to(text_embed.dtype)

        text_embed = torch.where(text_keep_mask, text_embed, null_text_embeds)

        # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

        learned_queries = repeat(self.learned_query, "d -> b 1 d", b=batch)

        # (b,n,d) in total
        # tokens = torch.cat((text_embed, learned_queries), dim=-2)
        tokens = torch.cat((text_encodings, text_embed, learned_queries), dim=-2)

        # attend

        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        # (b,d) finally
        pred_image_embed = tokens[..., -1, :]

        return self.l2norm_clamp_embed(pred_image_embed)


if __name__ == "__main__":
    from torchinfo import summary

    net = DiffusionPriorNetwork(
        depth=12,
        heads=12,
        ff_mult=4,
        attn_dropout=0.05,
        ff_dropout=0.05,
        normformer=True,
    )
    summary(net, input_size=[(7, 768), (7, 77, 768)])
