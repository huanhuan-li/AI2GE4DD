import os
import time
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf

import dataloader
from model import VAE2DD

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='/home/hli/vae2dd/data/L1000/sel_by_celltype_rand600_pertype', help='path to input data')
parser.add_argument('-fx', type=str, default='repX1_n13539X978_time6_xcp_train.npy')
parser.add_argument('-fd', type=str, default='repX1_n13539X978_time6_dcp_train.npy')
parser.add_argument('-fy', type=str, default='repX1_n13539X978_time6_ycp_train.npy')
parser.add_argument('-fxt', type=str, default='repX1_n3000X978_time6_xcp_val.npy')
parser.add_argument('-fdt', type=str, default='repX1_n3000X978_time6_dcp_val.npy')
parser.add_argument('-fyt', type=str, default='repX1_n3000X978_time6_ycp_val.npy')
parser.add_argument('-log_dir', default='/home/hli/vae2dd/result/log/')
parser.add_argument('-save_dir', default='/home/hli/vae2dd/result/')
parser.add_argument('-checkpoint_dir', default='checkpoint/')
parser.add_argument('-sample_dir', default='sample/')
parser.add_argument('-out', type=str, default=None, help='dir name')
parser.add_argument('-batchsize', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('-numepoch', type=int, default=400001, metavar='N', help='epoch')
parser.add_argument('-load_model', default=None, type=str)
parser.add_argument('-TEST_MODE', action='store_true', default=False, help='mode')
parser.add_argument('-TEST_OUT', type=int, default=None, help='global_step')
args = parser.parse_args()

'''settings'''
if args.out == None:
    model_name = "test_run" + "_" + str(np.random.randint(1000))
else:
    model_name = args.out
log_dir = args.log_dir + model_name
checkpoint_dir = args.save_dir + args.checkpoint_dir + model_name
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

sample_dir = args.save_dir + args.sample_dir + model_name
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

'''load data'''
input_param = {
    'path_to_x': os.path.join(args.d, args.fx),
    'path_to_d': os.path.join(args.d, args.fd),
    'path_to_y': os.path.join(args.d, args.fy),
    'batchsize': args.batchsize,
    'shuffle': True,
}
data_handler = dataloader.InputHandle(input_param)

input_param_val = {
    'path_to_x': os.path.join(args.d, args.fxt),
    'path_to_d': os.path.join(args.d, args.fdt),
    'path_to_y': os.path.join(args.d, args.fyt),
    'batchsize': args.batchsize,
    'shuffle': False,
}
data_val = dataloader.InputHandle(input_param_val)

'''load model'''
zdim = 50
xdim = ydim = data_handler.xdim
ddim = data_handler.ddim
learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999
mdl = VAE2DD(xdim = xdim,
                ddim = ddim, 
                ydim = ydim, 
                zdim = zdim,
                lr = learning_rate,
                beta1 = beta1,
                beta2 = beta2)

def train():
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    global_step=0

    for epoch in range(args.numepoch):
        batch_x, batch_d, batch_y = data_handler.samp_batch()
        mdl.train_one_step(batch_x=batch_x, batch_y=batch_y, batch_d=batch_d)
        if global_step%10000==0:
            with summary_writer.as_default():
                tf.summary.scalar('Loss/rec/x_rec_loss', mdl.x_rec_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/rec/d_rec_loss', mdl.d_rec_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/rec/y_rec_loss', mdl.y_rec_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/kl/x_kl_loss', mdl.x_kl_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/kl/d_kl_loss', mdl.d_kl_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/total_loss', mdl.total_loss_metric.result(), step=global_step)
        #save checkpoints
        if global_step % 100000 == 0:
            manager = tf.train.CheckpointManager(mdl.ckpt, str(checkpoint_dir + '/' + str(global_step)), max_to_keep=3)
            save_path = manager.save()
            
        global_step+=1

if args.TEST_MODE:
    test()
else:
    train()