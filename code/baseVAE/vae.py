# coding=utf-8

"""
TODO:
    Implementation of VAE based models for unsupervised learning of disentangled representations.
"""
import math
import tensorflow as tf

from utilies import batchnorm
import linear

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
    
class VAE2DD():
    """The designed vae2dd model."""
    def __init__(self, batchsize, xdim, ddim, ydim, zdim, lr, beta1, beta2):
        
        self.batchsize = batchsize
        self.xdim = xdim
        self.ddim = ddim
        self.ydim = ydim
        self.zdim = zdim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        
        #self.build()
        self.build_with_all_ctrl()

    def build(self):

        '''data flow'''
        # encode
        self.input_x = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        self.input_d = tf.placeholder(tf.float32, (self.batchsize, self.ddim))
        self.input_y = tf.placeholder(tf.float32, (self.batchsize, self.ydim))
        
        self.zx_mean, self.zx_logvar = self.encodeX(self.input_x, 
                                                    self.zdim, 
                                                    reuse=False, 
                                                    bn=True)
                                                    
        self.zd_mean, self.zd_logvar = self.encodeD(self.input_d, 
                                                    self.zdim, 
                                                    reuse=False, 
                                                    bn=True)
                                                                                    
        self.zx_sampled = self.sample_from_latent_distribution(self.zx_mean, self.zx_logvar)
        self.zd_sampled = self.sample_from_latent_distribution(self.zd_mean, self.zd_logvar)
        
        # decode
        self.x_rec = self.decodeX(self.zx_sampled, 
                                    self.xdim, 
                                    reuse=False, 
                                    bn=True)
                                    
        self.d_rec = self.decodeD(self.zd_sampled, 
                                    self.ddim, 
                                    reuse=False, 
                                    bn=True)
                                    
        self.zy_sampled = tf.multiply(self.zx_sampled, self.zd_sampled, name='zd_act_on_zx')
        self.y_rec = self.decodeX(self.zy_sampled, 
                                    self.ydim, 
                                    reuse=True, 
                                    bn=True)
        
        '''vars'''
        encodeX_vars = [var for var in tf.trainable_variables() if 'encodeX' in var.name]
        encodeD_vars = [var for var in tf.trainable_variables() if 'encodeD' in var.name]
        decodeX_vars = [var for var in tf.trainable_variables() if 'decodeX' in var.name]
        decodeD_vars = [var for var in tf.trainable_variables() if 'decodeD' in var.name]
        #decodeY_vars = [var for var in tf.trainable_variables() if 'decodeY' in var.name]
        
        '''losses'''
        self.x_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_x - self.x_rec), 1))
        #self.d_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_d - self.d_rec), 1))
        self.d_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_d, 
                                                                                               logits=self.d_rec), 1))
        self.y_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_y - self.y_rec), 1))
        
        self.x_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.zx_mean) + tf.exp(self.zx_logvar) - self.zx_logvar - 1, [1]), name="x_kl_loss")
        self.d_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.zd_mean) + tf.exp(self.zd_logvar) - self.zd_logvar - 1, [1]), name="d_kl_loss")
        
        # elbo
        self.loss = self.x_rec_loss + self.d_rec_loss + self.y_rec_loss + self.x_kl_loss + self.d_kl_loss
        
        '''Optimizers'''
        self.optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        with tf.name_scope("train_op"):
            self.train_step = self.optim.minimize(self.loss, var_list=encodeX_vars+encodeD_vars+decodeX_vars+decodeD_vars) #+decodeY_vars)
            
    def build_with_all_ctrl(self):

        '''data flow'''
        # encode
        self.input_x_ = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        self.input_x = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        self.input_d = tf.placeholder(tf.float32, (self.batchsize, self.ddim))
        self.input_y = tf.placeholder(tf.float32, (self.batchsize, self.ydim))
        
        self.zx_mean_, self.zx_logvar_ = self.encodeX(self.input_x_, 
                                                    self.zdim, 
                                                    reuse=False, 
                                                    bn=True)
        
        self.zx_mean, self.zx_logvar = self.encodeX(self.input_x, 
                                                    self.zdim, 
                                                    reuse=True, 
                                                    bn=True)
                                                    
        self.zd_mean, self.zd_logvar = self.encodeD(self.input_d, 
                                                    self.zdim, 
                                                    reuse=False, 
                                                    bn=True)
                                                    
        self.zx_sampled_ = self.sample_from_latent_distribution(self.zx_mean_, self.zx_logvar_)        
        self.zx_sampled = self.sample_from_latent_distribution(self.zx_mean, self.zx_logvar)
        self.zd_sampled = self.sample_from_latent_distribution(self.zd_mean, self.zd_logvar)
        
        # decode
        
        self.x_rec_ = self.decodeX(self.zx_sampled_, 
                                    self.xdim, 
                                    reuse=True, 
                                    bn=True)
                                    
        self.x_rec = self.decodeX(self.zx_sampled, 
                                    self.xdim, 
                                    reuse=False, 
                                    bn=True)
                                    
        self.d_rec = self.decodeD(self.zd_sampled, 
                                    self.ddim, 
                                    reuse=False, 
                                    bn=True)
                                    
        self.zy_sampled = tf.multiply(self.zx_sampled, self.zd_sampled, name='zd_act_on_zx')
        self.y_rec = self.decodeX(self.zy_sampled, 
                                    self.ydim, 
                                    reuse=True, 
                                    bn=True)
        
        '''vars'''
        encodeX_vars = [var for var in tf.trainable_variables() if 'encodeX' in var.name]
        encodeD_vars = [var for var in tf.trainable_variables() if 'encodeD' in var.name]
        decodeX_vars = [var for var in tf.trainable_variables() if 'decodeX' in var.name]
        decodeD_vars = [var for var in tf.trainable_variables() if 'decodeD' in var.name]
        #decodeY_vars = [var for var in tf.trainable_variables() if 'decodeY' in var.name]
        
        '''losses'''
        self.x_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_x_ - self.x_rec_), 1))
        #self.d_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_d - self.d_rec), 1))
        self.d_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_d, 
                                                                                               logits=self.d_rec), 1))
        self.y_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_y - self.y_rec), 1))
        
        self.x_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.zx_mean_) + tf.exp(self.zx_logvar_) - self.zx_logvar_ - 1, [1]), name="x_kl_loss_")
        self.d_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.zd_mean) + tf.exp(self.zd_logvar) - self.zd_logvar - 1, [1]), name="d_kl_loss")
        
        # elbo
        self.loss = self.x_rec_loss + self.d_rec_loss + self.y_rec_loss + self.x_kl_loss + self.d_kl_loss 
        
        '''Optimizers'''
        self.optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        with tf.name_scope("train_op"):
            self.train_step = self.optim.minimize(self.loss, var_list=encodeX_vars+encodeD_vars+decodeX_vars+decodeD_vars) #+decodeY_vars)
    
    def encodeX(self, input, num_latent, reuse, bn):
    
        with tf.variable_scope('encodeX', reuse=reuse) as scope:
        
            n_layers = {
                        'n_layer_1': 512,
                        'n_layer_2': 256,
                        'n_layer_3': 256,
            }
        
            output = input
            
            output = LeakyReLULayer('encodeX.0', self.xdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])    
            
            output = LeakyReLULayer('encodeX.1', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])
                
            output = LeakyReLULayer('encodeX.2', n_layers['n_layer_2'], n_layers['n_layer_3'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            mu = linear.Linear('encodeX.out.mu', n_layers['n_layer_3'], num_latent, output, initialization='he')
            log_var = linear.Linear('encodeX.out.logvar', n_layers['n_layer_3'], num_latent, output, initialization='he')

        return mu, log_var
        
    def encodeD(self, input, num_latent, reuse, bn):
    
        with tf.variable_scope('encodeD', reuse=reuse) as scope:
        
            n_layers = {
                        'n_layer_1': 512,
                        'n_layer_2': 256,
                        'n_layer_3': 256,
            }
        
            output = input
            
            output = LeakyReLULayer('encodeD.0', self.ddim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])    
            
            output = LeakyReLULayer('encodeD.1', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])
                
            output = LeakyReLULayer('encodeD.2', n_layers['n_layer_2'], n_layers['n_layer_3'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            mu = linear.Linear('encodeD.out.mu', n_layers['n_layer_3'], num_latent, output, initialization='he')
            log_var = linear.Linear('encodeD.out.logvar', n_layers['n_layer_3'], num_latent, output, initialization='he')

        return mu, log_var
        
    def sample_from_latent_distribution(self, z_mean, z_logvar):
        return tf.add(z_mean, tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1), name="sampled_latent_variable")
        
    def decodeX(self, input, out_dim, reuse, bn):
    
        with tf.variable_scope('decodeX', reuse=reuse) as scope:
    
            n_layers = {
                        'n_layer_1': 256,
                        'n_layer_2': 256,
                        'n_layer_3': 512,
            }
        
            output = input
        
            output = LeakyReLULayer('decodeX.0', self.zdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = LeakyReLULayer('decodeX.1', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])
                
            output = LeakyReLULayer('decodeX.2', n_layers['n_layer_2'], n_layers['n_layer_3'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = linear.Linear('decodeX.out', n_layers['n_layer_3'], out_dim, output, initialization='he')
            output = tf.nn.relu(output)
        
        return output
        
    def decodeD(self, input, out_dim, reuse, bn):
    
        with tf.variable_scope('decodeD', reuse=reuse) as scope:
    
            n_layers = {
                        'n_layer_1': 256,
                        'n_layer_2': 256,
                        'n_layer_3': 512,
            }
        
            output = input
        
            output = LeakyReLULayer('decodeD.0', self.zdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = LeakyReLULayer('decodeD.1', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])
                
            output = LeakyReLULayer('decodeD.2', n_layers['n_layer_2'], n_layers['n_layer_3'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = linear.Linear('decodeD.out', n_layers['n_layer_3'], out_dim, output, initialization='he')
            #output = tf.nn.relu(output)
        
        return output
    
    def decodeY(self, input, out_dim, reuse, bn):
    
        with tf.variable_scope('decodeY', reuse=reuse) as scope:
    
            n_layers = {
                        'n_layer_1': 256,
                        'n_layer_2': 256,
                        'n_layer_3': 512,
            }
        
            output = input
        
            output = LeakyReLULayer('decodeY.0', self.zdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = LeakyReLULayer('decodeY.1', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])
                
            output = LeakyReLULayer('decodeY.2', n_layers['n_layer_2'], n_layers['n_layer_3'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = linear.Linear('decodeY.out', n_layers['n_layer_3'], out_dim, output, initialization='he')
            output = tf.nn.relu(output)
        
        return output
