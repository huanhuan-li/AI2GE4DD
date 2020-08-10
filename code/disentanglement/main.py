# coding=utf-8

import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import vae
import utilies
import metrics
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='/home/hli/vae2dd/data/Microwell_MCA_imputation', help='path to input data')
parser.add_argument('-f', type=str, default='Microwell_fig2_forGAN_train.npy', help='input train data')
parser.add_argument('-fv', type=str, default='Microwell_fig2_forGAN_val.npy', help='input val data')
parser.add_argument('-ft', type=str, default='Microwell_fig2_forGAN_test.npy', help='input test data')
parser.add_argument('-log_dir', default='log/')
parser.add_argument('-checkpoint_dir', default='checkpoint/')
parser.add_argument('-sample_dir', default='sample/')
parser.add_argument('-save_dir', default='/home/hli/vae2dd/result/')
parser.add_argument('-out', type=str, default=None, help='dir name')
parser.add_argument('-batchsize', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('-numepoch', type=int, default=1000000, metavar='N', help='epoch')
parser.add_argument('-mode', type=str, default=None, help='disentangle vae type: {betaVAE, annealedVAE, betaTCVAE, factorVAE, DIPVAE}')
parser.add_argument('-hyperparam', '--hyperparam', nargs='+', type=int, default=[], help='hyper parameters setting for different disentabgle vae mode')
parser.add_argument('-load_model', default=None, type=str)
parser.add_argument('-TEST_MODE', action='store_true', default=False, help='mode')
parser.add_argument('-TEST_OUT', type=int, default=None, help='global_step')
parser.add_argument('-INTERVENE_MODE', action='store_true', default=False, help='mode type')
parser.add_argument('-intervention_factor_idxs', '--intervention_factor_idxs', 
                    nargs='+', type=int, default=[], help='latent dim which has capacity > cutoff')
parser.add_argument('-fqz', type=str, default=None, help='input qz data')
args = parser.parse_args()
print(args.hyperparam)

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

def shuffle_qz(z):
    z_shuffle = np.empty(shape=[z.shape[0], 0])
    for i in range(z.shape[1]):
        append_array = z[:, i]
        append_array_shuffled = np.random.permutation(append_array)
        z_shuffle = np.append(z_shuffle, append_array_shuffled.reshape(-1,1), axis=1)
    return z_shuffle

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
zdim = 50
xdim = data_test.xdim
learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999
if args.mode == 'betaVAE':
    mdl = vae.BetaVAE(batchsize = args.batchsize,
                    xdim = xdim,
                    zdim = zdim,
                    lr = learning_rate,
                    beta1 = beta1,
                    beta2 = beta2,
                    beta = args.hyperparam[0])
if args.mode == 'info-betaVAE':
    mdl = vae.infoBetaVAE(batchsize = args.batchsize,
                    xdim = xdim,
                    zdim = zdim,
                    lr = learning_rate,
                    beta1 = beta1,
                    beta2 = beta2,
                    beta = args.hyperparam[0],
                    gamma = args.hyperparam[1], 
                    c = args.hyperparam[2])
elif args.mode == 'annealedVAE':
    mdl = vae.AnnealedVAE(batchsize = args.batchsize,
                        xdim = xdim,
                        zdim = zdim,
                        lr = learning_rate,
                        beta1 = beta1,
                        beta2 = beta2,
                        gamma = 1., 
                        c_max = 1., 
                        iteration_threshold = 1e3)
elif args.mode == 'betaTCVAE':
    mdl = vae.BetaTCVAE(batchsize = args.batchsize,
                    xdim = xdim,
                    zdim = zdim,
                    lr = learning_rate,
                    beta1 = beta1,
                    beta2 = beta2,
                    beta = 1.0)
elif args.mode == 'factorVAE':
    mdl = vae.FactorVAE(batchsize = args.batchsize,
                    xdim = xdim,
                    zdim = zdim,
                    lr = learning_rate,
                    beta1 = beta1,
                    beta2 = beta2,
                    gamma = 1.0)
elif args.mode == 'DIPVAE':
    mdl = vae.DIPVAE(batchsize = args.batchsize,
                    xdim = xdim,
                    zdim = zdim,
                    lr = learning_rate,
                    beta1 = beta1,
                    beta2 = beta2,
                    lambda_od = 1.,
                    lambda_d_factor = 1.,
                    dip_type="i")

'''tensorboard monitor'''
#tf.summary.scalar('Loss', mdl.loss)
tf.summary.scalar('Loss/reconstruction_loss', mdl.reconstruction_loss)
tf.summary.scalar('Loss/kl_loss', mdl.kl_loss)
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
            
            if args.mode in ['betaVAE', 'info-betaVAE', 'betaTCVAE', 'DIPVAE']:
                feeds = {
                        mdl.input_x: batch_x,
                }
                _, summaries_ = sess.run([mdl.train_step, train_summaries], feed_dict=feeds)
            elif args.mode in ['annealedVAE']:
                feeds = {
                        mdl.input_x: batch_x,
                        mdl.global_step: global_step,
                }
                _, summaries_ = sess.run([mdl.train_step, train_summaries], feed_dict=feeds)
            elif args.mode in ['factorVAE']:
                feeds = {
                        mdl.input_x: batch_x,
                }
                _, _, summaries_ = sess.run([mdl.vae_train_step, mdl.dis_train_step, train_summaries], feed_dict=feeds)
                
            writer.add_summary(summaries_, global_step)
            
            if global_step % 10000 == 0:
                
                samp_x = np.empty(shape=[0, mdl.xdim])
                rec_x = np.empty(shape=[0, mdl.xdim])
                val_zmean = np.empty(shape=[0, mdl.zdim])
                val_zlogvar = np.empty(shape=[0, mdl.zdim])
                val_qz = np.empty(shape=[0, mdl.zdim])

                # generate sample
                val_size = 0
                for i in range(data_val.N//args.batchsize):
                    
                    batch_x = data_val.samp_batch()
                    
                    if args.mode in ['annealedVAE']:
                        feeds = {
                                mdl.input_x: batch_x,
                                mdl.global_step: global_step,
                        }
                    else:
                        feeds = {
                                mdl.input_x: batch_x,
                        }
                    px, zmean, zlogvar, qz = sess.run([mdl.reconstructions, mdl.z_mean, mdl.z_logvar, mdl.z_sampled], feed_dict=feeds)
                    samp_x = np.append(samp_x, batch_x, axis=0)
                    rec_x = np.append(rec_x, px, axis=0)
                    val_zmean = np.append(val_zmean, zmean, axis=0)
                    val_zlogvar = np.append(val_zlogvar, zlogvar, axis=0)
                    val_qz = np.append(val_qz, qz, axis=0)
                    
                    val_size += args.batchsize
                
                # calcu eval
                ## reconstructions
                gene_sim_mu = np.mean(utilies.calcu_rsquare_distance(samp_x, rec_x))
                cosine_sim_mu, _ = cosine_sim(samp_x, rec_x, moments=True)
                
                ## disentangle
                val_pz = np.random.normal(size=[val_size, mdl.zdim])
                mmd_qz_pz = metrics.compute_mmd(val_qz, val_pz) # numeric
                val_qz_ = shuffle_qz(val_qz)
                mmd_qz_qz_ = metrics.compute_mmd(val_qz, val_qz_) # numeric
                qz_capacity_array = metrics.comput_capacity(val_zmean, val_zlogvar) # array, len=zdim
                
                ## write to summary
                val_summaries = tf.Summary()
                distribute_sim = [gene_sim_mu, cosine_sim_mu, mmd_qz_pz, mmd_qz_qz_]
                for i, t in enumerate(['Similarity/gene_sim_mu',
                                        'Similarity/samp_sim_mu',
                                        'Disentangle/mmd_qz_pz',
                                        'Disentangle/mmd_qz_qz_']):
                    val_summaries.value.add(tag=t, simple_value=distribute_sim[i])
                writer.add_summary(val_summaries, global_step)
                log_histogram(writer, 'Disentangle/qz_capacity_hist', qz_capacity_array, global_step, bins=100)
            
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
        
        qz_zmean = np.empty(shape=[0, mdl.zdim])
        qz_zlogvar = np.empty(shape=[0, mdl.zdim])
        qz_samp = np.empty(shape=[0, mdl.zdim])
        rec_x = np.empty(shape=[0, mdl.xdim])
        qz_zmean_from_recx = np.empty(shape=[0, mdl.zdim])
        qz_zlogvar_from_recx = np.empty(shape=[0, mdl.zdim])

        for i in range(data_test.N//args.batchsize):
            
            batch_x = data_test.samp_batch()
            zmean_, zlogvar_ = mdl.encoder(tf.convert_to_tensor(batch_x, dtype=tf.float32), mdl.zdim, reuse=True, bn=True)
            qz_ = tf.add(zmean_, tf.exp(zlogvar_ / 2) * tf.random_normal(tf.shape(zmean_), 0, 1))
            px_ = mdl.decoder(qz_, mdl.xdim, reuse=True, bn=True)
            zmean_from_recx_, zlogvar_from_recx_ = mdl.encoder(px_, mdl.zdim, reuse=True, bn=True)
            
            zmean, zlogvar, qz, px, zmean_from_recx, zlogvar_from_recx = sess.run([zmean_, zlogvar_, qz_, px_, zmean_from_recx_, zlogvar_from_recx_])
            
            qz_zmean = np.append(qz_zmean, zmean, axis=0)
            qz_zlogvar = np.append(qz_zlogvar, zlogvar, axis=0)
            qz_samp = np.append(qz_samp, qz, axis=0)
            rec_x = np.append(rec_x, px, axis=0)
            qz_zmean_from_recx = np.append(qz_zmean_from_recx, zmean_from_recx, axis=0)
            qz_zlogvar_from_recx = np.append(qz_zlogvar_from_recx, zlogvar_from_recx, axis=0)
            
        # save
        np.save(os.path.join(sample_dir, 'zmean_step_{}.npy'.format(args.TEST_OUT)), qz_zmean)
        np.save(os.path.join(sample_dir, 'zlogvar_step_{}.npy'.format(args.TEST_OUT)), qz_zlogvar)
        np.save(os.path.join(sample_dir, 'qz_step_{}.npy'.format(args.TEST_OUT)), qz_samp)
        np.save(os.path.join(sample_dir, 'reconstructions_step_{}.npy'.format(args.TEST_OUT)), rec_x)
        np.save(os.path.join(sample_dir, 'zmean_from_recx_step_{}.npy'.format(args.TEST_OUT)), qz_zmean_from_recx)
        np.save(os.path.join(sample_dir, 'zlogvar_from_recx_step_{}.npy'.format(args.TEST_OUT)), qz_zlogvar_from_recx)
        
'''factor intervention'''
def intervene():

    intervention_dir = sample_dir + '/intervention'
    if not os.path.exists(intervention_dir):
        os.mkdir(intervention_dir)
    
    intervened_steps=1000
    #intervened_factor_idxs = args.intervention_factor_idxs
    test_qz = np.load(os.path.join(sample_dir, args.fqz))
    intervened_qz_1 = tf.placeholder(tf.float32, (intervened_steps, mdl.zdim), name='intervened_qz_1')
    intervened_qz_2 = tf.placeholder(tf.float32, (test_qz.shape[0], mdl.zdim), name='intervened_qz_2')

    saver=tf.train.Saver()
    configProt = tf.ConfigProto()
    configProt.gpu_options.allow_growth = True
    configProt.allow_soft_placement = True
    
    with tf.Session(config=configProt) as sess:

        if args.load_model is not None:
            saver.restore(sess=sess, save_path=args.load_model)
        else:
            print("InputError: model should exist in test mode!")
            
        #### construct arrays which target dim is varied and other dims keep the same
        output1 = []
        keep_qz = np.zeros(shape=(intervened_steps, mdl.zdim))
        #for samp_idx in [-1, 0, 50, 100, 200, 500, 1000, 2000]:
        #keep_qz = np.expand_dims(test_qz[samp_idx,:], 0).repeat(intervened_steps, axis=0)
        for intervened_factor_idx in range(mdl.zdim):        
            target_array = test_qz.copy()[:, intervened_factor_idx]
            intervened_array = np.linspace(target_array.min(), target_array.max(), num=intervened_steps)
            intervened_qz_ = keep_qz.copy()
            intervened_qz_[:, intervened_factor_idx] = intervened_array
                
            px_after_intervened_qz = sess.run(mdl.decoder(intervened_qz_1, mdl.xdim, reuse=True, bn=True), 
                                                feed_dict={intervened_qz_1: intervened_qz_})
            output1.append(px_after_intervened_qz)
            
        np.save(os.path.join(intervention_dir, 'reconstructions_after_intervention_target_dim.npy'), output1)
        del px_after_intervened_qz, output1, keep_qz, intervened_qz_
        
        #### construct arrays which target dim keep the same (is 0) and other dims keep varied
        output2 = []
        df_ = pd.DataFrame(columns=['gene_sim_mu', 'samp_cosine_sim_mu'])
        # perturb and generate samples
        for d in range(mdl.zdim):
            print(d)
            intervened_qz_ = test_qz.copy()
            intervened_qz_[:,d] = 0
            #name='zdim_{}'.format(d)
            gen_samps = sess.run(mdl.decoder(intervened_qz_2, mdl.xdim, reuse=True, bn=True), 
                                    feed_dict={intervened_qz_2: intervened_qz_})
            output2.append(gen_samps)
            
            # calc similarity
            #gene_sim_mu = np.mean(utilies.calcu_rsquare_distance(gen_samps, data_test.data[0:test_qz.shape[0], :]))
            #samp_cosine_sim_mu, _ = cosine_sim(gen_samps, data_test.data[0:test_qz.shape[0], :], moments=True)
            #mmd_dis = metrics.compute_mmd(gen_samps, data_test.data[0:test_qz.shape[0], :])
            #df_ = df_.append(pd.Series({'gene_sim_mu': gene_sim_mu, 
            #                            'samp_cosine_sim_mu': samp_cosine_sim_mu}, name=name))
        #df_.to_csv(os.path.join(intervention_dir, 'samp_similarity_after_intervention_each_dim.csv'))
        np.save(os.path.join(intervention_dir, 'reconstructions_after_intervention_other_dim.npy'), output2)
        
if args.TEST_MODE:
    test()
elif args.INTERVENE_MODE:
    intervene()
else:
    train()