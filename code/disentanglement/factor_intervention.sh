#! /bin/sh

source /home/cudasoft/python349/env.sh
export CUDA_VISIBLE_DEVICES=7

# test mode
set_name=VAE_datasetb_s1_beta_is10
step=800000
intervention_factor_idxs="8 18 31 33"
python /home/hli/vae2dd/code/main.py 	-out ${set_name} \
										-mode betaVAE \
										-hyperparam "1" \
										-INTERVENE_MODE \
										-load_model "/home/hli/vae2dd/result/checkpoint/${set_name}/${step}/step_${step}.ckpt" \
										-intervention_factor_idxs $intervention_factor_idxs \
										-fqz 'qz_step_800000.npy'
				