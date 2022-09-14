import torch
from kornia.losses import total_variation
from torch import autograd, nn


def get_general_cost(energy_type):
    cost_dict = {
        "mse": mse_loss,
        "mse_and_TV": mse_tv_loss,
        "ip_loss": ip_loss,
        "discriminator_cost": discriminator_cost,
    }
    assert energy_type in cost_dict
    return cost_dict[energy_type]


def ip_loss(tensor1, tensor2, coeff_ip=1.0):
    # * c(x,y)=<x,y> / dim(x)
    return -coeff_ip * (tensor1 * tensor2).mean()


def mse_loss(tensor1, tensor2, coeff_mse=1.0):
    # * Here it's normalized quadratic cost:
    # * c(x,y)=||x-y||^2 / dim(x)
    return coeff_mse * (tensor1 - tensor2).pow(2).mean()


def discriminator_cost(fake_img, discriminator):
    # disctimniator gives high score to true images,
    # low score to fake images
    disc_loss = -discriminator(fake_img).mean()
    return disc_loss


def mse_tv_loss(tensor1, tensor2, mask, coeff_tv):
    l2_loss = mse_loss(tensor1, tensor2 * (1 - mask))
    tv_loss = total_variation(tensor2 * mask).mean()
    return l2_loss + coeff_tv * tv_loss


def gradientOptimality(discriminator, fake_images, source_images, coeff_go=0.0):
    if not coeff_go:
        return 0.0
    # fake_images = nn.Parameter(fake_images, requires_grad=True)
    fake_images.requires_grad = True
    f_tx = discriminator(fake_images)

    gradients = autograd.grad(
        outputs=f_tx,
        inputs=fake_images,
        grad_outputs=torch.ones(f_tx.size()).to(fake_images.device),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_penalty = (gradients - source_images).mean(dim=0).norm("fro")
    return coeff_go * grad_penalty


def gradientPenalty(discriminator, real_images, fake_images, coeff_gp=0.0):
    if not coeff_gp:
        return 0.0
    batch_size = real_images.shape[0]
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(
        batch_size, real_images.size(1), real_images.size(2), real_images.size(3)
    )
    eta = eta.to(real_images.device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    interpolated = interpolated.to(real_images.device)

    interpolated = nn.Parameter(interpolated, requires_grad=True)

    prob_interpolated = discriminator(interpolated)

    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(real_images.device),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return coeff_gp * grad_penalty
