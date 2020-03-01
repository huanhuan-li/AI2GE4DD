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
    
def gaussian_log_density(samples, mean, log_var):
    pi = tf.constant(math.pi)
    normalization = tf.log(2. * pi)
    inv_var = tf.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_var + log_var + normalization)
    
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

class BaseVAE:
    """The basic Gaussian encoder model."""
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2):
        
        self.batchsize = batchsize
        self.xdim = xdim
        self.zdim = zdim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def build(self):

        '''data flow'''
        self.input_x = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        
        z_mean, z_logvar = self.encoder(self.input_x, 
                                        self.zdim, 
                                        reuse=False, 
                                        bn=True)
        
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        
        self.reconstructions = self.decoder(z_sampled, 
                                        self.xdim, 
                                        reuse=False, 
                                        bn=True)
        
        '''vars'''
        encoder_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        decoder_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        
        '''losses (elbo with regularizer)'''
        self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_x - self.reconstructions), 1)) # need check
        kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]), name="kl_loss") # need check
        # elbo
        elbo = tf.add(self.reconstruction_loss, kl_loss, name="elbo")
        # losses
        self.regularizer_ = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
        self.loss = tf.add(self.reconstruction_loss, self.regularizer_, name="loss")
        
        '''Optimizers'''
        self.optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        with tf.name_scope("train_op"):
            self.train_step = self.optim.minimize(self.loss, var_list=encoder_vars+decoder_vars)
            
    def encoder(self, input, num_latent, reuse, bn):
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
        
    def decoder(self, input, out_dim, reuse, bn):
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
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2, beta):
        
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.beta = beta
        super().build()
    
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        return self.beta * kl_loss

def total_correlation(z, z_mean, z_logvar):
    """
    Estimate of total correlation on a batch. 
    Args:
        z: [batch_size, num_latents]-tensor with sampled representation.
        z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
        z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
        Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = gaussian_log_density(samples = tf.expand_dims(z, 1),
                                        mean = tf.expand_dims(z_mean, 0),
                                        log_var = tf.expand_dims(z_logvar, 0))
                                        
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = tf.reduce_logsumexp(tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
                                                axis=1, keepdims=False)
                                                
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i))) + constant) 
    # for each sample in the batch, which is a vector of size [batch_size,]. 
    log_qz_product = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
                                                axis=1, keepdims=False)
    return tf.reduce_mean(log_qz - log_qz_product)
    
class BetaTCVAE(BaseVAE):
    """BetaTCVAE model."""
    # https://arxiv.org/pdf/1802.04942
    # If alpha = gamma = 1, Eq.4 can be written as ELBO + (1 - beta) * TC.
    
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2, beta):       
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.beta = beta
        super().build()
        
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
        return tc + kl_loss
        
class FactorVAE(BaseVAE):
    """FactorVAE model. https://arxiv.org/pdf/1802.05983"""
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2, gamma):       
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.gamma = gamma
        
    def fc_discriminator(input_tensor, reuse):
        """Fully connected discriminator used in FactorVAE paper for all datasets.
        Args:
            input_tensor: Input tensor of shape (None, num_latents) to build discriminator on.
        Returns:
            logits: Output tensor of shape (batch_size, 2) with logits from discriminator.
            probs: Output tensor of shape (batch_size, 2) with probabilities from discriminator.
        """
        with tf.variable_scope('fc_discriminator', reuse=reuse) as scope:
        
            n_layers = {
                        'n_layer_1': 128,
                        'n_layer_2': 128,
            }
        
            output = tf.layers.flatten(input_tensor)
        
            output = LeakyReLULayer('fc_discriminator.1', self.zdim, n_layers['n_layer_1'], output)
        
            output = LeakyReLULayer('fc_discriminator.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
        
            logits = linear.Linear('fc_discriminator.3', n_layers['n_layer_2'], 2, output, initialization='he')
            probs = tf.nn.softmax(logits)
            return logits, tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
        
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        # regularizer_ = kl_loss + gamma * tc_loss
        # tc_loss = 
        z_shuffle = shuffle_codes(z_sampled)
        logits_z, probs_z = self.fc_discriminator(z_sampled, reuse=False)
        _, probs_z_shuffle = self.fc_discriminator(z_shuffle, reuse=True)
        tc_loss_per_sample = logits_z[:, 0] - logits_z[:, 1]
        tc_loss = tf.reduce_mean(tc_loss_per_sample, axis=0)
        return kl_loss + self.gamma * tc_loss
        
    def build(self):
        '''data flow'''
        self.input_x = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        z_mean, z_logvar = self.encoder(self.input_x, 
                                        self.zdim, 
                                        reuse=False, 
                                        bn=True)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        self.reconstructions = self.decoder(z_sampled, 
                                        self.xdim, 
                                        reuse=False, 
                                        bn=True)
        
        '''losses (elbo with regularizer)'''
        self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_x - self.reconstructions), 1)) # need check
        kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]), name="kl_loss") # need check
        # elbo
        elbo = tf.add(self.reconstruction_loss, kl_loss, name="elbo")
        # losses
        self.regularizer_ = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
        self.loss = tf.add(self.reconstruction_loss, self.regularizer_, name="loss")
        
        '''vars'''
        encoder_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        decoder_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        dis_vars = [var for var in tf.trainable_variables() if 'fc_discriminator' in var.name]
        
        '''Optimizers'''
        self.vae_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        self.dis_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        with tf.name_scope("train_op"):
            self.vae_train_step = self.optim.minimize(self.loss, var_list=encoder_vars+decoder_vars)
            self.dis_train_step = self.optim.minimize(self.loss, var_list=dis_vars)