# Our conditional map
python run.py name=fmnist2mnist_class_preserving datamodule.source.dataset=FMNIST datamodule.target.dataset=MNIST experiment=nist mode=paper trainer.max_epochs=140 logger=wandb
# model.coeff_mse=1e2 model.coeff_label=1e0 or 1e2 is the best quality.

python run.py name=mnist2kmnist_class_preserving datamodule.source.dataset=MNIST datamodule.target.dataset=KMNIST experiment=nist mode=paper trainer.max_epochs=70 logger=wandb

# Our unconditional map, this needs to be run after conditional map is finished!!
python run.py name=fit_uncond_fmnist2mnist_class_preserving datamodule.source.dataset=FMNIST datamodule.target.dataset=MNIST experiment=fit_class_map model.canonical_map_save_path="$(pwd)/logs/reproduce/fmnist2mnist_class_preserving/map_136_ema.pth" mode=paper logger=wandb

python run.py name=fit_uncond_mnist2kmnist_class_preserving datamodule.source.dataset=MNIST datamodule.target.dataset=KMNIST experiment=fit_class_map model.canonical_map_save_path="$(pwd)/logs/reproduce/mnist2kmnist_class_preserving/map_60_ema.pth" mode=paper logger=wandb
