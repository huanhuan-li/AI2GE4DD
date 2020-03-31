#! /bin/bash

python samp_embed_2types.py \
    -path_px /home/hli/vae2dd/result/sample/betaVAE_b_is_1 \
    -file_px reconstructions_step_600000.npy \
    -path_test /home/hli/vae2dd/data/Microwell_MCA \
    -file_test Microwell_fig2_forGAN_test.npy \
    -n_pcs 50\
    -path_save /home/hli/vae2dd/eval/output/betaVAE_b_is_1\
    -o step_600k\
    -HP_FLAG