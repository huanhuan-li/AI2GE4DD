# coding=utf-8

import os
import time
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf

import vae
import utilies
import metrics
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='/home/hli/vae2dd/data/L1000/sel_by_celltype_rand600_pertype', help='path to input data')
parser.add_argument('-fx', type=str, default='repX1_n13539X978_time6_xcp_train.npy')
parser.add_argument('-fd', type=str, default='repX1_n13539X978_time6_dcp_train.npy')
parser.add_argument('-fy', type=str, default='repX1_n13539X978_time6_ycp_train.npy')
parser.add_argument('-fxt', type=str, default='repX1_n13539X978_time6_xcp_val.npy')
parser.add_argument('-fdt', type=str, default='repX1_n13539X978_time6_dcp_val.npy')
parser.add_argument('-fyt', type=str, default='repX1_n13539X978_time6_ycp_val.npy')
parser.add_argument('-fxa', type=str, default='repX1_n1306X978_time6_all_xcp.npy')
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

def kl_divergence(p, p_hat, epsilon=1e-6):
    return p * np.log(p+epsilon) - p * np.log(p_hat+epsilon) + (1 - p) * np.log(1- p+epsilon) - (1 - p) * np.log(1-p_hat+epsilon)

def log_histogram(writer, tag, values, step, bins=100):
    # values is a numpy array
    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)
    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))
    bin_edges = bin_edges[1:]
    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)
    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()
    
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

input_param_4all_ctrl = {
    'path': os.path.join(args.d, args.fxa),
    'batchsize': args.batchsize,
    'shuffle': True,
}
data_all_ctrl = dataloader.InputHandle_(input_param_4all_ctrl)

all_drug = np.load('/home/hli/vae2dd/data/L1000/LINCS_GSE92742_20335_drug_fingerprint.npy')

'''load model'''
zdim = 50
xdim = ydim = data_handler.xdim
ddim = data_handler.ddim
learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999

mdl = vae.VAE2DD(batchsize = args.batchsize,
                xdim = xdim,
                ddim = ddim, 
                ydim = ydim, 
                zdim = zdim,
                lr = learning_rate,
                beta1 = beta1,
                beta2 = beta2)

'''tensorboard monitor'''
tf.summary.scalar('Loss/rec/x_rec_loss', mdl.x_rec_loss)
tf.summary.scalar('Loss/rec/d_rec_loss', mdl.d_rec_loss)
tf.summary.scalar('Loss/rec/y_rec_loss', mdl.y_rec_loss)

tf.summary.scalar('Loss/kl/x_kl_loss', mdl.x_kl_loss)
tf.summary.scalar('Loss/kl/d_kl_loss', mdl.d_kl_loss)

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
            
            batch_x_ = data_all_ctrl.samp_batch()
            batch_x, batch_d, batch_y = data_handler.samp_batch()
            
            feeds = {
                mdl.input_x: batch_x,
                mdl.input_x_: batch_x_,
                mdl.input_d: batch_d,
                mdl.input_y: batch_y,
                }
            _, summaries_ = sess.run([mdl.train_step, train_summaries], feed_dict=feeds)
            writer.add_summary(summaries_, global_step)
            
            if global_step % 10000 == 0:
                
                samp_x = np.empty(shape=[0, mdl.xdim])
                rec_x = np.empty(shape=[0, mdl.xdim])
                
                samp_d = np.empty(shape=[0, mdl.ddim])
                rec_d = np.empty(shape=[0, mdl.ddim])
                
                samp_y = np.empty(shape=[0, mdl.ydim])
                rec_y = np.empty(shape=[0, mdl.ydim])
                
                drug2y_corr = np.empty(shape=[0, len(all_drug)])

                # generate sample for similarity analysis
                val_size = 0
                for i in range(data_val.N//args.batchsize):
                    
                    batch_x, batch_d, batch_y = data_val.samp_batch()
                    
                    feeds = {
                            mdl.input_x: batch_x,
                            mdl.input_x_: batch_x,
                            mdl.input_d: batch_d,
                            mdl.input_y: batch_y,
                        }
                    px, pd, py = sess.run([mdl.x_rec, mdl.d_rec, mdl.y_rec], feed_dict=feeds)
                    samp_x = np.append(samp_x, batch_x, axis=0)
                    rec_x = np.append(rec_x, px, axis=0)
                    samp_d = np.append(samp_d, batch_d, axis=0)
                    rec_d = np.append(rec_d, pd, axis=0)
                    samp_y = np.append(samp_y, batch_y, axis=0)
                    rec_y = np.append(rec_y, py, axis=0)
                    
                    val_size += args.batchsize                    
                    
                if global_step % 100000 == 0:
                
                    ## drug score rank
                    all_drug_sub = np.arange(len(all_drug))
                    np.random.shuffle(all_drug_sub)
                    
                    curr_drug2y_corr = []
                    val_input_x_4rank = tf.placeholder(tf.float32, (val_size, mdl.xdim))
                    val_input_d_4rank = tf.placeholder(tf.float32, (val_size, mdl.ddim))
                    val_input_y_4rank = tf.placeholder(tf.float32, (val_size, mdl.ydim))
                    
                    val_zx_mu, val_zx_logvar = mdl.encodeX(val_input_x_4rank, mdl.zdim, reuse=True, bn=True)
                    val_zd_mu, val_zd_logvar = mdl.encodeD(val_input_d_4rank, mdl.zdim, reuse=True, bn=True)
                    val_zx_sampled = mdl.sample_from_latent_distribution(val_zx_mu, val_zx_logvar)
                    val_zd_sampled = mdl.sample_from_latent_distribution(val_zd_mu, val_zd_logvar)
                    val_zy_sampled = tf.multiply(val_zx_sampled, val_zd_sampled, name='zd_act_on_zx')
                    val_y_rec = mdl.decodeX(val_zy_sampled, mdl.ydim, reuse=True, bn=True)
                    
                    for idx,d in enumerate(all_drug[all_drug_sub[0:1000], :]):
                        
                        curr_d = np.expand_dims(d, 0).repeat(val_size, axis=0)
                            
                        feeds_4rank = {
                            val_input_x_4rank: samp_x,
                            val_input_d_4rank: curr_d,
                            val_input_y_4rank: samp_y, 
                            }
                                
                        py_ = sess.run(val_y_rec, feed_dict=feeds_4rank)
                        
                        corr = [stats.pearsonr(u,v)[0] for u,v in zip(py_, samp_y)]#list, len=batchsize
                        curr_drug2y_corr.append(corr)
                        
                    drug2y_corr = np.array(curr_drug2y_corr).T
                    
                    ## drug rank score
                    drug2y_corr_truth = [stats.pearsonr(u,v)[0] for u,v in zip(samp_y, rec_y)]
                    drug_rank_hist = [sum(drug2y_corr[i,:]>drug2y_corr_truth[i]) for i in range(val_size)]
                
                # calcu eval
                ## reconstructions
                x_sim_mu, _ = cosine_sim(samp_x, rec_x) #np.mean(utilies.calcu_rsquare_distance(samp_x, rec_x))
                y_sim_mu, _ = cosine_sim(samp_y, rec_y) #np.mean(utilies.calcu_rsquare_distance(samp_y, rec_y))
                #d_sim_mu, _ = cosine_sim(samp_d, rec_d) 
                
                ## write to summary
                val_summaries = tf.Summary()
                distribute_sim = [x_sim_mu, y_sim_mu]
                for i, t in enumerate(['Similarity/x_sim_mu',
                                        'Similarity/y_sim_mu',]):
                    val_summaries.value.add(tag=t, simple_value=distribute_sim[i])
                writer.add_summary(val_summaries, global_step)
                
                if global_step % 100000 == 0:
                    log_histogram(writer, 'Drug_rank/drug_rank_among_all', np.array(drug_rank_hist), global_step, bins=100)
            
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

        
        val_size = data_val.N
        val_input_x_4rank = tf.placeholder(tf.float32, (val_size, mdl.xdim))
        val_input_d_4rank = tf.placeholder(tf.float32, (val_size, mdl.ddim))
        val_input_y_4rank = tf.placeholder(tf.float32, (val_size, mdl.ydim))
        
        val_zx_mu, val_zx_logvar = mdl.encodeX(val_input_x_4rank, mdl.zdim, reuse=True, bn=True)
        val_zd_mu, val_zd_logvar = mdl.encodeD(val_input_d_4rank, mdl.zdim, reuse=True, bn=True)
        val_zx_sampled = mdl.sample_from_latent_distribution(val_zx_mu, val_zx_logvar)
        val_zd_sampled = mdl.sample_from_latent_distribution(val_zd_mu, val_zd_logvar)
        val_zy_sampled = tf.multiply(val_zx_sampled, val_zd_sampled+1, name='zd_act_on_zx')
        val_y_rec = mdl.decodeY(val_zy_sampled, mdl.ydim, reuse=True, bn=True)
        
        feeds_4rec = {
                    val_input_x_4rank: data_val.x_data,
                    val_input_d_4rank: data_val.d_data,
                    val_input_y_4rank: data_val.y_data, 
                    }
                                
        rec_y = sess.run(val_y_rec, feed_dict=feeds_4rec)
        
        curr_drug2y_corr = []
        processed_drugs = []          
        for idx,d in enumerate(all_drug):
                       
            curr_d = np.expand_dims(d, 0).repeat(val_size, axis=0)
            feeds_4rank = {
                        val_input_x_4rank: data_val.x_data,
                        val_input_d_4rank: curr_d,
                        val_input_y_4rank: data_val.y_data, 
                        }
                                
            py_ = sess.run(val_y_rec, feed_dict=feeds_4rank)
                        
            corr = [stats.pearsonr(u,v)[0] for u,v in zip(py_, data_val.y_data)]#list, len=batchsize
            curr_drug2y_corr.append(corr), processed_drugs.append(d)
                        
        drug2y_corr = np.array(curr_drug2y_corr).T
        ## drug rank score
        #drug2y_corr_truth = [stats.pearsonr(u,v)[0] for u,v in zip(data_val.y_data, rec_y)]
        #drug_rank_hist = [sum(drug2y_corr[i,:]>drug2y_corr_truth[i]) for i in range(val_size)]

        ## save
        np.save(os.path.join(sample_dir, 'drug2y_corr_step{}.npy'.format(args.TEST_OUT)), drug2y_corr)
        np.save(os.path.join(sample_dir, 'val_rec_y_step{}.npy'.format(args.TEST_OUT)), rec_y)
        
if args.TEST_MODE:
    test()
else:
    train()
