# coding=utf-8

"""
TODO:
    Implementation of VAE based models for unsupervised learning of disentangled representations.
"""

import tensorflow as tf
from utilies import batchnorm


def ReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(
        name, 
        n_in, 
        n_out, 
        inputs,
        initialization='he',
    )
    return tf.nn.relu(output)
    
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)
    
def LeakyReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(
        name, 
        n_in, 
        n_out, 
        inputs,
        initialization='he',
    )
    return LeakyReLU(output)

class BaseVAE(Object):
    """The basic Gaussian encoder model."""
    def __init__(batchsize,
                xdim,
                zdim,
                lr):
        self.batchsize = batchsize
        self.xdim = xdim
        self.zdim = zdim
        self.lr = lr
    
    def model_fn(self):

        '''data flow'''
        self.input_x = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        z_mean, z_logvar = self.encoder(self.input_x, 
                                        self.zdim, 
                                        reuse=False, 
                                        bn=True)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        reconstructions = self.decoder(z_sampled, 
                                        self.xdim, 
                                        reuse=False, 
                                        bn=True)
        
        '''vars'''
        encoder_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        decoder_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        
        '''losses (elbo with regularizer)'''
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
            
    def encoder(input, num_latent, reuse, bn):
        '''fully connected encoder'''
        with tf.variable_scope('encoder', reuse=reuse) as scope:
        
            n_layers = {
                        'n_layer_1': 512,
                        'n_layer_2': 512,
            }
        
            output = input
            
            output = LeakyReLULayer('encoder.1', self.xdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])    
            
            output = LeakyReLULayer('encoder.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])     
            
            mu = linear.Linear('encoder.out.mu', n_layers['n_layer_2'], num_latent, output, initialization='he')
            log_var = linear.Linear('encoder.out.var', n_layers['n_layer_2'], num_latent, output, initialization='he')

        return mu, log_var
        
    def sample_from_latent_distribution(self, z_mean, z_logvar):
        return tf.add(z_mean, tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1), name="sampled_latent_variable")
        
    def decoder(input, out_dim, reuse, bn):
        '''fully connected decoder'''
        with tf.variable_scope('decoder', reuse=reuse) as scope:
    
            n_layers = {
                        'n_layer_1': 512,
                        'n_layer_2': 512,
            }
        
            output = input
        
            output = LeakyReLULayer('decoder.1', self.zdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = LeakyReLULayer('decoder.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])    
            
            output = linear.Linear('decoder.3', n_layers['n_layer_2'], out_dim, output, initialization='he')
            output = tf.nn.relu(output)
        
        return output

class BetaVAE(BaseVAE):
    '''BetaVAE: https://openreview.net/forum?id=Sy2fzU9gl'''
    def __init__(self, beta):
        self.beta = beta
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        return self.beta * kl_loss
