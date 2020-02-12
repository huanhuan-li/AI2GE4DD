# coding=utf-8

import os
import argparse
import tensorflow as tf

import vae
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='/home/hli/vae2dd/data', help='path to input data')
parser.add_argument('-f', type=str, default='61389_merged_sample_12328_gene_time6_celltype8_xcp_expr.hdf5', help='input data file')
parser.add_argument('-log_dir', default='log/')
parser.add_argument('-checkpoint_dir', default='checkpoint/')
parser.add_argument('-sample_dir', default='sample/')
parser.add_argument('-save_dir', default='/home/hli/vae2dd/result/')
parser.add_argument('-out', type=str, default=None, help='dir name')
parser.add_argument('-batchsize', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('-numepoch', type=int, default=1000000, metavar='N', help='epoch')
args = parser.parse_args()

'''settings'''
if args.out == None:
    model_name = "test_run" + "_" + str(np.random.randint(1000))
else:
    model_name = args.out
log_dir = args.save_dir + args.log_dir + model_name
checkpoint_dir = args.save_dir + args.checkpoint_dir + model_name
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

sample_dir = args.save_dir + args.sample_dir + model_name
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

'''load data'''
input_param = {
    'path': os.path.join(args.d, args.f),
    'batchsize': args.batchsize,
    'shuffle': True,
}
data_handler = dataloader.InputHandle(input_param)
print(data_handler.xdim)

'''load model'''
mdl = vae.BetaVAE(
            batchsize = args.batchsize,
            xdim = data_handler.xdim,
            zdim = 50,
            lr = 1e-4,
            beta1 = 0.,
            beta2 = 0.9,
            beta = 1.0)

'''tensorboard monitor'''
tf.summary.scalar('Loss', mdl.loss)
train_summaries = tf.summary.merge_all()

'''train'''
def train():
    
    saver=tf.train.Saver()
    configProt = tf.ConfigProto()
    configProt.gpu_options.allow_growth = True
    configProt.allow_soft_placement = True
    
    with tf.Session(config=configProt) as sess:
        
        writer = tf.summary.FileWriter(str(log_dir), sess.graph)
        tf.global_variables_initializer().run()
        global_step = 0
        
        for epoch in range(args.numepoch):
        
            batch_x = data_handler.samp_batch()
            feeds = {
                    mdl.input_x: batch_x,
            }
            _, summaries_ = sess.run([mdl.train_step, train_summaries], feed_dict=feeds)
            writer.add_summary(summaries_, global_step)
            
            global_step += 1

'''test'''
# to be added

if __name__=='__main__':
    train()