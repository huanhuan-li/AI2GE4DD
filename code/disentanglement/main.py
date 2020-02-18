# coding=utf-8

import os
import argparse
import numpy as np
import tensorflow as tf

import vae
import utilies
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='/home/hli/vae2dd/data', help='path to input data')
parser.add_argument('-f', type=str, default='61295_merged_gene12328_time6_cell8_ycp_train.npy', help='input train data')
parser.add_argument('-fv', type=str, default='61295_merged_gene12328_time6_cell8_ycp_val.npy', help='input val data')
parser.add_argument('-ft', type=str, default='61295_merged_gene12328_time6_cell8_ycp_test.npy', help='input test data')
parser.add_argument('-log_dir', default='log/')
parser.add_argument('-checkpoint_dir', default='checkpoint/')
parser.add_argument('-sample_dir', default='sample/')
parser.add_argument('-save_dir', default='/home/hli/vae2dd/result/')
parser.add_argument('-out', type=str, default=None, help='dir name')
parser.add_argument('-batchsize', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('-numepoch', type=int, default=2000000, metavar='N', help='epoch')
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
    
'''functions for monitor'''
def cosine_sim(x, y, moments=True):
    x_val = np.sqrt(np.sum(np.multiply(x,x), axis=1))
    y_val = np.sqrt(np.sum(np.multiply(y,y), axis=1))
    
    normmat = np.multiply(x_val.reshape(x_val.shape[0], -1), y_val)
    innerprod = np.matmul(x, y.T)
    consinemat = innerprod/normmat

    consinedis = np.mean(consinemat, axis=1)
    if moments==True:
        return np.mean(consinedis), np.std(consinedis)
    else:
        return consinedis

'''load data'''
input_param = {
    'path': os.path.join(args.d, args.f),
    'batchsize': args.batchsize,
    'shuffle': True,
}
data_handler = dataloader.InputHandle(input_param)

input_param_val = {
    'path': os.path.join(args.d, args.fv),
    'batchsize': args.batchsize,
    'shuffle': False,
}
data_val = dataloader.InputHandle(input_param_val)

input_param_test = {
    'path': os.path.join(args.d, args.ft),
    'batchsize': args.batchsize,
    'shuffle': False,
}
data_test = dataloader.InputHandle(input_param_test)

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
#tf.summary.scalar('Loss', mdl.loss)
tf.summary.scalar('Loss/reconstruction_loss', mdl.reconstruction_loss)
tf.summary.scalar('Loss/regularizer_', mdl.regularizer_)
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
            
            if global_step % 10000 == 0:
                
                samp_x = np.empty(shape=[0, mdl.xdim])
                rec_x = np.empty(shape=[0, mdl.xdim])

                # generate sample
                for i in range(data_val.N//args.batchsize):
                    
                    batch_x = data_val.samp_batch()
                    feeds = {
                            mdl.input_x: batch_x,
                    }
                    px = sess.run(mdl.reconstructions, feed_dict=feeds)
                    samp_x = np.append(samp_x, batch_x, axis=0)
                    rec_x = np.append(rec_x, px, axis=0)
                
                # calcu eval
                gene_sim_mu = np.mean(utilies.calcu_rsquare_distance(samp_x, rec_x))
                cosine_sim_mu, _ = cosine_sim(samp_x, rec_x, moments=True)
                val_summaries = tf.Summary()
                distribute_sim = [gene_sim_mu, cosine_sim_mu]
                for i, t in enumerate(['Similarity/gene_sim_mu',
                                        'Similarity/samp_sim_mu']):
                    val_summaries.value.add(tag=t, simple_value=distribute_sim[i])
                writer.add_summary(val_summaries, global_step)
            
            # save model
            if global_step % 200000 == 0:
                saver.save(sess, os.path.join(str(checkpoint_dir + '/' + str(global_step)), 'step_{}.ckpt'.format(global_step)))
            
            global_step += 1

'''test'''
def test():

    saver=tf.train.Saver()

    with tf.Session() as sess:

        if args.load_model is not None:
            saver.restore(sess=sess, save_path=args.load_model)
        else:
            print("InputError: model should exist in test mode!")
        
        rec_x = np.empty(shape=[0, mdl.xdim])

        for i in range(data_test.N//args.batchsize):
            batch_x = data_test.samp_batch()
            feeds = {
                mdl.input_x: batch_x,
            }
            px = sess.run(mdl.reconstructions, feed_dict=feeds)
            rec_x = np.append(rec_x, px, axis=0)
        
        # save
        np.save(os.path.join(sample_dir, 'reconstructions_step_{}.npy'.format(args.TEST_OUT)), rec_x)

if __name__=='__main__':
    train()