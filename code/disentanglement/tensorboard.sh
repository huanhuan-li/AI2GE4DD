#! /bin/sh

#$ -S /bin/bash
#$ -q p100
#$ -l ngpus=1
#$ -hard -l hostname=comput34.ghddi.org
#$ -cwd
#$ -e ./err_log
#$ -o ./out_log
#$ -N ts

source /home/cudasoft/python349/env.sh

source /home/cudasoft/bin/startcuda.sh
# echo $CUDA_VISIBLE_DEVICES
tensorboard --logdir=/home/hli/vae2dd/result/log --port=8008
source /home/cudasoft/bin/end_cuda.sh
