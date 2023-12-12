# cc3m
python run.py name=reproduce_full_cc3m experiment=text2img datamodule.sub_ds_path="cc3m_test/cc3m_no_watermark" mode=paper logger=wandb

# laion-art
python run.py name=reproduce_full_laion-art experiment=text2img datamodule.sub_ds_path="laion-art_test/laion-high-resolution-en" datamodule.num_train_data=3e5  datamodule.start_shard=323 datamodule.end_shard=328 mode=paper logger=wandb

# test
python run.py name=recover_test_cc3m experiment=text2img mode=test_txt2img callbacks.sampling.emb_map_path=$(pwd)/logs/reproduce/reproduce_full_cc3m/map_6_ema.pth run_folder=$(pwd)/logs/reproduce/reproduce_full_cc3m
# The generated results are in logs/test/recover_test_cc3m

python run.py name=recover_test_laion experiment=text2img mode=test_txt2img datamodule.sub_ds_path="laion_art/laion-high-resolution-en" datamodule.sub_ds_path="laion_art/laion-high-resolution-en" datamodule.num_train_data=3e5  datamodule.start_shard=323 datamodule.end_shard=328 model.coeff_mse=5e5 callbacks.sampling.emb_map_path="$(pwd)/logs/reproduce/reproduce_full_laion-art/map_6_ema.pth" run_folder=$(pwd)/logs/reproduce/reproduce_full_laion-art

# The generated results are in logs/test/recover_test_laion
