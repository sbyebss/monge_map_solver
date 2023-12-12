python run.py name=mse_128_resolution experiment=inpainting_celeb128 model.n_inner_iter=10 mode=paper trainer.devices="[0]" logger=wandb

python run.py name=mse_64_resolution experiment=inpainting_celeb64 model.n_inner_iter=5 mode=paper trainer.devices="[0]" logger=wandb
