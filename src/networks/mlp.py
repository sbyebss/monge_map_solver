from dalle2_pytorch.dalle2_pytorch import l2norm
from torch import nn


class ResNet(nn.Module):
    def __init__(
        self,
        feat_dim=2,
        output_dim=1,
        hidden_dim=1024,
        num_layer=1,
    ):
        super().__init__()
        self.layer1 = nn.Linear(feat_dim, hidden_dim)
        self.layer1_activ = nn.PReLU()
        self.linearblock = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layer)]
        )
        self.resblock = nn.ModuleList(
            [nn.Linear(feat_dim, hidden_dim) for _ in range(num_layer)]
        )
        self.atvt_list = nn.ModuleList([nn.PReLU() for _ in range(num_layer)])
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, feat):
        input_feat = feat
        x = self.layer1_activ(self.layer1(feat))
        for i, layer in enumerate(self.linearblock):
            x = self.atvt_list[i](layer(x)) + self.resblock[i](input_feat)

        x = self.last_layer(x)
        return x


class ResDiscriminator(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=1024, num_layer=1):
        super().__init__()
        self.disc = ResNet(feat_dim, 1, hidden_dim, num_layer)

    def forward(self, feat):
        return self.disc(feat)


class ResGenerator(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=1024, num_layer=1):
        super().__init__()
        self.generator = ResNet(feat_dim, feat_dim, hidden_dim, num_layer)
        self.image_embed_scale = feat_dim**0.5

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    # pylint: disable=unused-argument
    def forward(self, text_embed=None, text_encodings=None):
        text_embed = self.l2norm_clamp_embed(text_embed)
        output_emb = self.generator(text_embed)
        return self.l2norm_clamp_embed(output_emb)


if __name__ == "__main__":
    from torchinfo import summary

    snet = ResDiscriminator(num_layer=3)
    summary(snet, input_size=[7, 768])
