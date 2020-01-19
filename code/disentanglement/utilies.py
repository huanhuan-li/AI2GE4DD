# coding=utf-8

import tensorflow as tf

def compute_gaussian_kl(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]), name="kl_loss")
    
def anneal(c_max, step, iteration_threshold):
    """Anneal function for anneal_vae https://arxiv.org/abs/1804.03599
    Args:
        c_max: Maximum capacity.
        step: Current step.
        iteration_threshold: How many iterations to reach c_max.
    Returns:
        Capacity annealed linearly until c_max."""
    return tf.math.minimum(c_max * 1., c_max * 1. * tf.to_float(step) / iteration_threshold)