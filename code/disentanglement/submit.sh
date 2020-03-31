#! /bin/sh

#$ -S /bin/bash
#$ -q p100
#$ -l ngpus=1
#$ -hard -l hostname=comput33.ghddi.org
#$ -cwd
#$ -e ./out/err_log
#$ -o ./out/out_log
#$ -N gct_3

source /home/cudasoft/python349/env.sh

source /home/cudasoft/bin/startcuda.sh
#export CUDA_VISIBLE_DEVICES=7
#echo $CUDA_VISIBLE_DEVICES >> qsub.log
# train mode
python ggan_CE_tcell_set2_main.py -WGAN WGAN-GP -out ggan_CE_tcell_set3
# test mode
#python ggan_tcell_19_main.py -WGAN WGAN-GP \
#				-out ggan_tcell_set19 \
#				-TEST_OUT 600000 \
#				-TEST_MODE \
#				-load_model '/home/hli/GAN/graphGAN_tf/out/checkpoint/ggan_tcell_set19/600000/step_600000.ckpt'
source /home/cudasoft/bin/end_cuda.sh
