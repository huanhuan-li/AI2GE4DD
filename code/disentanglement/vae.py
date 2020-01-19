# coding=utf-8

"""
TODO:
    Implementation of VAE based models for unsupervised learning of disentangled representations.
    Adapted from: https://github.com/google-research/disentanglement_lib
"""

import tensorflow as tf
import utilies
import losses
import optimizers
import architectures

class GaussianEncoderModel(object):
    """Abstract base class of a Gaussian encoder model."""
    
    def model_fn(self, features, labels, mode, params):
        raise NotImplementedError()
        
    def gaussian_encoder(self, input_tensor, is_training):
        """Applies the Gaussian encoder to L1000.
        Args:
            input_tensor: Tensor with the observations to be encoded.
            is_training: Boolean indicating whether in training mode."""
        raise NotImplementedError()
        
    def decode(self, latent_tensor, observation_shape, is_training):
        raise NotImplementedError()
        
    def sample_from_latent_distribution(self, z_mean, z_logvar):
        """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
        return tf.add(z_mean, tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1), name="sampled_latent_variable")
  
class BaseVAE(GaussianEncoderModel):
    """The basic Gaussian encoder model."""
    def model_fn(self, input, labels, mode, params):
        """Args:
        model: 'train' or not;"""
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        data_shape = 
        z_mean, z_logvar = self.gaussian_encoder(input, is_training=is_training)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        reconstructions = self.decode(z_sampled, data_shape, is_training)
        # item 1 in elbo
        per_sample_loss = losses.make_reconstruction_loss(input, reconstructions)
        reconstruction_loss = tf.reduce_mean(per_sample_loss)
        # item 2 in elbo
        kl_loss = utilies.compute_gaussian_kl(z_mean, z_logvar)
        elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
        # loss
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
        loss = tf.add(reconstruction_loss, regularizer, name="loss")
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = optimizers.make_vae_optimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            train_op = tf.group([train_op, update_ops])
            tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            tf.summary.scalar("elbo", -elbo)
            
    def gaussian_encoder(self, input_tensor, is_training):
        return architectures.make_gaussian_encoder(input_tensor, 
                                                        is_training=is_training)
        
    def decode(self, latent_tensor, observation_shape, is_training):
        return architectures.make_decoder(latent_tensor, 
                                                observation_shape, 
                                                is_training=is_training)

class BetaVAE(BaseVAE):
    '''BetaVAE: https://openreview.net/forum?id=Sy2fzU9gl'''
    def __init__(self, beta):
        self.beta = beta
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        return self.beta * kl_loss

class AnnealedVAE(BaseVAE):
    '''AnnealedVAE model: https://arxiv.org/abs/1804.03599'''
    def __init__(self, gamma, c_max, iteration_threshold):
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold
        
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        c = utilies.anneal(self.c_max, tf.train.get_global_step(), self.iteration_threshold)
        return self.gamma * tf.math.abs(kl_loss - c)

class FactorVAE(BaseVAE):
    '''FactorVAE: https://arxiv.org/pdf/1802.05983'''
    def __init__(self, gamma):
        self.gamma = gamma
    
    def model_fn(self, input, labels, mode, params):
        del labels
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        data_shape = 
        z_mean, z_logvar = self.gaussian_encoder(input, is_training=is_training)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        z_shuffle = shuffle_codes(z_sampled)