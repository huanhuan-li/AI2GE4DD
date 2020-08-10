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
        
        self.z_mean, self.z_logvar = self.encoder(self.input_x, 
                                        self.zdim, 
                                        reuse=False, 
                                        bn=True)
        
        self.z_sampled = self.sample_from_latent_distribution(self.z_mean, self.z_logvar)
        
        self.reconstructions = self.decoder(self.z_sampled, 
                                        self.xdim, 
                                        reuse=False, 
                                        bn=True)
                                        
        self.z_mean_, self.z_logvar_ = self.encoder(self.reconstructions,  # encode from reconstruction sample
                                        self.zdim, 
                                        reuse=True, 
                                        bn=True)
        
        '''vars'''
        encoder_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        decoder_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        
        '''losses (elbo with regularizer)'''
        self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_x - self.reconstructions), 1)) 
        self.kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.exp(self.z_logvar) - self.z_logvar - 1, [1]), name="kl_loss")
        # elbo
        elbo = tf.add(self.reconstruction_loss, self.kl_loss, name="elbo")
        # losses
        self.regularizer_ = self.regularizer(self.kl_loss, self.z_mean, self.z_logvar, self.z_sampled, self.z_mean_, self.z_logvar_)
        self.loss = tf.add(self.reconstruction_loss, self.regularizer_, name="loss")
        
        '''Optimizers'''
        self.optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        with tf.name_scope("train_op"):
            self.train_step = self.optim.minimize(self.loss, var_list=encoder_vars+decoder_vars)
            
    def encoder(self, input, num_latent, reuse, bn):
        '''fully connected encoder'''
        with tf.variable_scope('encoder', reuse=reuse) as scope:
        
            n_layers = {
                        'n_layer_1': 1024,
                        'n_layer_2': 512,
                        'n_layer_3': 512,
            }
        
            output = input
            
            output = LeakyReLULayer('encoder.1', self.xdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])    
            
            output = LeakyReLULayer('encoder.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])
                
            output = LeakyReLULayer('encoder.3', n_layers['n_layer_2'], n_layers['n_layer_3'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            mu = linear.Linear('encoder.out.mu', n_layers['n_layer_3'], num_latent, output, initialization='he')
            log_var = linear.Linear('encoder.out.var', n_layers['n_layer_3'], num_latent, output, initialization='he')

        return mu, log_var
        
    def sample_from_latent_distribution(self, z_mean, z_logvar):
        return tf.add(z_mean, tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1), name="sampled_latent_variable")
        
    def decoder(self, input, out_dim, reuse, bn):
        '''fully connected decoder'''
        with tf.variable_scope('decoder', reuse=reuse) as scope:
    
            n_layers = {
                        'n_layer_1': 512,
                        'n_layer_2': 512,
                        'n_layer_3': 1024,
            }
        
            output = input
        
            output = LeakyReLULayer('decoder.1', self.zdim, n_layers['n_layer_1'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = LeakyReLULayer('decoder.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
            if bn:
                output = batchnorm(output, axis=[0])
                
            output = LeakyReLULayer('decoder.3', n_layers['n_layer_2'], n_layers['n_layer_3'], output)
            if bn:
                output = batchnorm(output, axis=[0])
            
            output = linear.Linear('decoder.4', n_layers['n_layer_3'], out_dim, output, initialization='he')
            output = tf.nn.relu(output)
        
        return output

class BetaVAE(BaseVAE):
    '''BetaVAE: https://openreview.net/forum?id=Sy2fzU9gl'''
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2, beta):
        
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.beta = beta
        super().build()
    
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled, z_mean_, z_logvar_):
        del z_mean, z_logvar, z_sampled, z_mean_, z_logvar_
        return self.beta * kl_loss
        
def calcElogN(qz, z_mean_, z_logvar_):
    return tf.reduce_mean( -0.5*z_logvar_ - tf.square(qz - z_mean_)/2*tf.exp(z_logvar_), 0) # give a result of dimension [zdim]
    
class infoBetaVAE(BaseVAE):
    '''self designed'''
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2, beta, gamma, c):
        
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.beta = beta
        self.gamma = gamma
        self.c = c
        super().build()
    
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled, z_mean_, z_logvar_):
        del z_mean, z_logvar
        mi = calcElogN(z_sampled, z_mean_, z_logvar_)
        return self.beta * kl_loss + self.gamma * tf.reduce_mean(tf.abs(mi - self.c))
        
def anneal(c_max, step, iteration_threshold):
    """Anneal function for anneal_vae (https://arxiv.org/abs/1804.03599).
    Args:
        c_max: Maximum capacity of the bottleneck.
        step: Current step.
        iteration_threshold: How many iterations to reach c_max.
    Returns:
        Capacity annealed linearly until c_max.
    """
    return tf.minimum(c_max * 1., c_max * 1. * tf.to_float(step) / iteration_threshold)
    
class AnnealedVAE(BaseVAE):
    """AnnealedVAE model."""
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2, gamma, c_max, iteration_threshold):
        '''
        Args:
            gamma: Hyperparameter for the regularizer.
        '''
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold
        super().build()
        
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        self.global_step = tf.placeholder(tf.int32, shape=[], name='global_step')
        c = anneal(self.c_max, self.global_step, self.iteration_threshold)
        return self.gamma * tf.abs(kl_loss - c)
        
def gaussian_log_density(samples, mean, log_var):
    pi = tf.constant(math.pi)
    normalization = tf.log(2. * pi)
    inv_var = tf.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_var + log_var + normalization)

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

class FactorVAE(BaseVAE):
    """FactorVAE model. https://arxiv.org/pdf/1802.05983"""
    def __init__(self, batchsize, xdim, zdim, lr, beta1, beta2, gamma):       
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.gamma = gamma
        self.build()
        
    def fc_discriminator(self, input_tensor, reuse):
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
            probs = tf.nn.softmax(logits) # softmax per sample
            return logits, tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
        
    def build(self):
        '''data flow'''
        self.input_x = tf.placeholder(tf.float32, (self.batchsize, self.xdim))
        z_mean, z_logvar = self.encoder(self.input_x, 
                                        self.zdim, 
                                        reuse=False, 
                                        bn=True)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        z_shuffle = shuffle_codes(z_sampled)
        self.reconstructions = self.decoder(z_sampled, 
                                        self.xdim, 
                                        reuse=False, 
                                        bn=True)
                                        
        '''Discriminator'''
        logits_z, probs_z = self.fc_discriminator(z_sampled, reuse=False)
        _, probs_z_shuffle = self.fc_discriminator(z_shuffle, reuse=True)
        
        '''losses (elbo with regularizer)'''
        ### factorVAE loss
        self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_x - self.reconstructions), 1)) # need check
        self.kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]), name="kl_loss") # need check
        # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
        tc_loss = tf.reduce_mean(logits_z[:, 0] - logits_z[:, 1], axis=0)
        self.regularizer_ = self.kl_loss + self.gamma * tc_loss
        self.loss = tf.add(self.reconstruction_loss, self.regularizer_, name="factorVAE_loss")
        
        ### discriminator loss
        self.discr_loss = tf.add(0.5 * tf.reduce_mean(tf.log(probs_z[:, 0])),
                                0.5 * tf.reduce_mean(tf.log(probs_z_shuffle[:, 1])),
                                name="discriminator_loss")
        
        '''vars'''
        encoder_vars = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        decoder_vars = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        dis_vars = [var for var in tf.trainable_variables() if 'fc_discriminator' in var.name]
        
        '''Optimizers'''
        self.vae_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        self.dis_optim = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        with tf.name_scope("train_op"):
            self.vae_train_step = self.vae_optim.minimize(self.loss, var_list=encoder_vars+decoder_vars)
            self.dis_train_step = self.dis_optim.minimize(self.discr_loss, var_list=dis_vars)
            
def compute_covariance_z_mean(z_mean):
    """Computes the covariance of z_mean: cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.
    Args:
        z_mean: Encoder mean, tensor of size [batch_size, num_latent].
    Returns:
        cov_z_mean: Covariance of encoder mean, tensor of size [num_latent, num_latent].
    """
    expectation_z_mean_z_mean_t = tf.reduce_mean(tf.expand_dims(z_mean, 2) * tf.expand_dims(z_mean, 1), axis=0)
    expectation_z_mean = tf.reduce_mean(z_mean, axis=0)
    cov_z_mean = tf.subtract(expectation_z_mean_z_mean_t,
                            tf.expand_dims(expectation_z_mean, 1) * tf.expand_dims(expectation_z_mean, 0))
    return cov_z_mean
    
def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
    """Compute on and off diagonal regularizers for DIP-VAE models. Penalize deviations of covariance_matrix from the identity matrix.
    Args:
        covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
        lambda_od: Weight of penalty for off diagonal elements.
        lambda_d: Weight of penalty for diagonal elements.
    Returns:
        dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
    """
    covariance_matrix_diagonal = tf.diag_part(covariance_matrix)
    covariance_matrix_off_diagonal = covariance_matrix - tf.diag(covariance_matrix_diagonal)
    dip_regularizer = tf.add(lambda_od * tf.reduce_sum(covariance_matrix_off_diagonal**2),
                            lambda_d * tf.reduce_sum((covariance_matrix_diagonal - 1)**2))
    return dip_regularizer

class DIPVAE(BaseVAE):
    """DIPVAE model. https://arxiv.org/pdf/1711.00848.pdf"""
    def __init__(self, 
                batchsize, 
                xdim, 
                zdim, 
                lr, 
                beta1, 
                beta2,
                lambda_od,
                lambda_d_factor,
                dip_type="i"):
        '''
        Args:
            lambda_od: Hyperparameter for off diagonal values of covariance matrix.
            lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
            lambda_d = lambda_d_factor*lambda_od.
            dip_type: "i" or "ii".
        '''
        super().__init__(batchsize, xdim, zdim, lr, beta1, beta2)
        self.lambda_od = lambda_od
        self.lambda_d_factor = lambda_d_factor
        self.dip_type = dip_type
        super().build()
        
    
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        cov_z_mean = compute_covariance_z_mean(z_mean)
        lambda_d = self.lambda_d_factor * self.lambda_od
        if self.dip_type == "i":
            # mu = z_mean is [batch_size, num_latent]
            # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
            cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z_mean, self.lambda_od, lambda_d)
        elif self.dip_type == "ii":
            cov_enc = tf.matrix_diag(tf.exp(z_logvar))
            expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
            cov_z = expectation_cov_enc + cov_z_mean
            cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z, self.lambda_od, lambda_d)
        else:
            raise NotImplementedError("DIP variant not supported.")
        
        return kl_loss + cov_dip_regularizer

