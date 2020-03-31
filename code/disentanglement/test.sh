#! /bin/sh

source /home/cudasoft/python349/env.sh
export CUDA_VISIBLE_DEVICES=7

# test mode
python main.py 	-out betaVAE_b_is_1 \
				-mode betaVAE \
				-hyperparam "1" \
				-TEST_OUT 600000 \
				-TEST_MODE \
				-load_model '/home/hli/vae2dd/result/checkpoint/betaVAE_b_is_1/600000/step_600000.ckpt'
