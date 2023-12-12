import torch

# import torch.nn.functional as F
from einops import repeat
from torch import nn
from torch.nn import Linear, Module, ModuleList

# Model from https://github.com/luost26/diffusion-point-cloud


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class Generator(Module):
    def __init__(self, context_dim=256, residual=True):
        super().__init__()
        self.act = nn.PReLU()
        self.residual = residual
        self.layers = ModuleList(
            [
                ConcatSquashLinear(3, 128, context_dim),
                ConcatSquashLinear(128, 256, context_dim),
                ConcatSquashLinear(256, 512, context_dim),
                ConcatSquashLinear(512, 256, context_dim),
                ConcatSquashLinear(256, 128, context_dim),
                ConcatSquashLinear(128, 3, context_dim),
            ]
        )
        self.context_dim = context_dim

    def forward(self, x):
        """
        x:  Point clouds (B, N, d).
        """
        batch_size = x.size(0)
        context = torch.randn(self.context_dim).to(x.device)
        ctx_emb = repeat(context, "d -> b 1 d", b=batch_size)  # (B, 1, F)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class Discriminator(Module):
    def __init__(self, context_dim=256, residual=True):
        super().__init__()
        self.act = nn.PReLU()
        self.residual = residual
        self.layers = ModuleList(
            [
                ConcatSquashLinear(3, 128, context_dim),
                # ConcatSquashLinear(128, 256, context_dim),
                # ConcatSquashLinear(256, 512, context_dim),
                # ConcatSquashLinear(512, 256, context_dim),
                # ConcatSquashLinear(256, 128, context_dim),
                ConcatSquashLinear(128, 3, context_dim),
            ]
        )
        self.context = nn.Parameter(torch.randn(context_dim))

    def forward(self, x):
        """
        x:  Point clouds (B, N, d).
        """
        batch_size = x.size(0)
        ctx_emb = repeat(self.context, "d -> b 1 d", b=batch_size)  # (B, 1, F)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return out.sum()
        else:
            return (out - x).sum()


# class Discriminator(Module):
#     def __init__(self, context_dim=256, residual=True):
#         super().__init__()
#         self.net = Generator(context_dim=context_dim, residual=residual)

#     def forward(self, x):
#         """
#         Args:
#             x:  Point clouds (B, N, d).
#         """
#         return (self.net(x) - x).sum()


if __name__ == "__main__":
    from torchinfo import summary

    net = Generator()
    summary(net, input_size=(7, 2048, 3))

    net = Discriminator()
    summary(net, input_size=(7, 2048, 3))
