#! /bin/bash

#$ -S /bin/bash
#$ -q rocket
#$ -pe rocket 1
#$ -cwd
#$ -e err_log
#$ -o out_log
#$ -N eval

source activate vae2dd
python eval.py \
	-real /home/hli/vae2dd/result/sample/baseVAE_replicate/val_realy_mu_step200000.npy \
	-pred /home/hli/vae2dd/result/sample/baseVAE_replicate/val_geny_mu_step200000.npy \
	-out_path /home/hli/vae2dd/eval/output/baseVAE_no_affine_coupling \
	-out_name eval_metrics_step200000.csv