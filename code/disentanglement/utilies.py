# coding=utf-8

import numpy as np
import tensorflow as tf
from scipy.stats import percentileofscore

def compute_gaussian_kl(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]), name="kl_loss")
    
def batchnorm(inputs, axis, scale=None, offset=None, variance_epsilon=0.001, name=None):
    with tf.variable_scope('batchnorm'):
        mean, var = tf.nn.moments(inputs, axis, keep_dims=True)
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, variance_epsilon, name=name)
    return result
    
def anneal(c_max, step, iteration_threshold):
    """Anneal function for anneal_vae https://arxiv.org/abs/1804.03599
    Args:
        c_max: Maximum capacity.
        step: Current step.
        iteration_threshold: How many iterations to reach c_max.
    Returns:
        Capacity annealed linearly until c_max."""
    return tf.math.minimum(c_max * 1., c_max * 1. * tf.to_float(step) / iteration_threshold)
    
def shuffle_codes(z):
    """Shuffles latent variables across the batch.
    Args:
        z: [batch_size, num_latent] representation.
    Returns:
        shuffled: [batch_size, num_latent] shuffled representation across the batch.
    """
    z_shuffle = []
    for i in range(z.get_shape()[1]):
        z_shuffle.append(tf.random_shuffle(z[:, i]))
        shuffled = tf.stack(z_shuffle, 1, name="latent_shuffled")
    return shuffled
    
def calcu_rsquare_distance(ref_, samp_):
    result = []
    for gene_idx in range(ref_.shape[1]):
        ref = ref_[:, gene_idx]
        samp = samp_[:, gene_idx]
        
        sample = [0.02] + [i for i in np.arange(0.05, 1.0, 0.05)]
        
        samp_pct_x = []
        samp_pct_y = []
        
        for i,s in enumerate(sample):
            # theoretical quantiles
            samp_pct_x.append(percentileofscore(ref, s))
            # sample quantiles
            samp_pct_y.append(percentileofscore(samp, s))
        
        # estimated quantile distance
        r = np.linalg.norm(np.subtract(np.asarray(samp_pct_x)/100, np.asarray(samp_pct_y)/100), ord=2)
        result.append(r)
    return result