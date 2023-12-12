# pylint: disable=line-too-long,too-many-instance-attributes,too-many-locals,unused-import
from dalle2_pytorch.dalle2_pytorch import CausalTransformer
from einops.layers.torch import Rearrange
from torch import nn


class Discriminator(nn.Module):
    def __init__(
        self,
        dim=768,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
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

        # self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim=dim, **kwargs)

    def forward(self, text_embed=None):
        input_text_embed = text_embed
        text_embed = self.to_text_embeds(text_embed)

        # batch, _ = text_embed.shape
        # learned_queries = repeat(self.learned_query, "d -> b 1 d", b=batch)
        # tokens = torch.cat((text_embed, learned_queries), dim=-2)
        tokens = text_embed

        # attend
        tokens = self.causal_transformer(tokens)
        pred_image_embed = tokens[..., -1, :]

        return (pred_image_embed - input_text_embed).sum()


if __name__ == "__main__":
    from torchinfo import summary

    net = Discriminator(
        depth=4,
        heads=4,
        ff_mult=4,
        attn_dropout=0.05,
        ff_dropout=0.05,
        normformer=True,
    )
    summary(net, input_size=[(7, 768)])
