import os

from jammy.utils.git import git_rootdir
from torchvision import transforms
from torchvision.datasets import CIFAR10, KMNIST, MNIST, USPS, FashionMNIST

from .celeba import CelebA

__all__ = ["get_img_dataset"]


def nist_dataset(dataset, data_path, img_size=32):
    if dataset == "MNIST":
        torch_dataset = MNIST
    elif dataset == "USPS":
        torch_dataset = USPS
    elif dataset == "FMNIST":
        torch_dataset = FashionMNIST
    elif dataset == "KMNIST":
        torch_dataset = KMNIST
    train = torch_dataset(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        ),
    )
    test = torch_dataset(
        data_path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        ),
    )
    return train, test


def init_data_config(config):
    if "path" not in config:
        config.path = git_rootdir("data")


def get_img_dataset(config):  # pylint: disable=too-many-branches
    init_data_config(config)
    if config.dataset in ["MNIST", "USPS", "FMNIST", "KMNIST"]:
        return nist_dataset(config.dataset, config.path, config.image_size)
    if config.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )

    if config.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(config.path, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(config.path, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.dataset == "CELEBA":
        if config.random_flip:
            dataset = CelebA(
                root=os.path.join(config.path, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(config.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=False,
            )
        else:
            dataset = CelebA(
                root=os.path.join(config.path, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(config.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=False,
            )

        test_dataset = CelebA(
            root=os.path.join(config.path, "datasets", "celeba_test"),
            split="test",
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(140),
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=False,
        )

    return dataset, test_dataset
