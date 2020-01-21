# coding=utf-8

"""
TODO:
    Implementation of VAE based models for unsupervised learning of disentangled representations.
"""

import tensorflow as tf
import utilies
import architectures

class BaseVAE(Object):
    """The basic Gaussian encoder model."""
    def __init__(batchsize,
                xdim,
                lr):
        self.batchsize = batchsize
        self.xdim = xdim
        self.lr = lr
    
    def model_fn(self):

        '''data flow'''
        self.input_x = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        z_mean, z_logvar = self.gaussian_encoder(self.input_x, is_training=is_training)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        reconstructions = self.decoder(z_sampled, data_shape, is_training)
        
        '''vars'''
        encoder_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        decoder_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        
        '''elbo and losses (elbo with regularizer)'''
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_x - reconstructions), 1)) # need check
        kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]), name="kl_loss") # need check
        # elbo
        elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
        # losses
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
        loss = tf.add(reconstruction_loss, regularizer, name="loss")
        
        '''Optimizers'''
        self.optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        with tf.name_scope("train_op"):
            self.train_step = self.e_optim.minimize(self.loss, var_list=encoder_vars+decoder_vars)
            
    def gaussian_encoder(self, input_tensor, is_training):
        return architectures.make_gaussian_encoder(input_tensor, is_training=is_training)
        
    def decoder(self, latent_tensor, observation_shape, is_training):
        return architectures.make_decoder(latent_tensor, observation_shape, is_training=is_training)
        
    def sample_from_latent_distribution(self, z_mean, z_logvar):
        return tf.add(z_mean, tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1), name="sampled_latent_variable")

class BetaVAE(BaseVAE):
    '''BetaVAE: https://openreview.net/forum?id=Sy2fzU9gl'''
    def __init__(self, beta):
        self.beta = beta
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        return self.beta * kl_loss
