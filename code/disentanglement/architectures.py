# coding=utf-8

"""Library of commonly used architectures and reconstruction losses."""

import numpy as np
import tensorflow as tf
import linear

def make_gaussian_encoder(input_tensor,
                                num_latent,
                                encoder_fn,
                                is_training=True):
    '''Args:
        num_latent: Integer with dimensionality of latent space.
        encoder_fn: Function that that takes the arguments (input_tensor, num_latent, is_training) and returns the tuple (means, log_vars) 
        '''
    with tf.variable_scope("encoder"):
        return encoder_fn(input_tensor=input_tensor,
                            num_latent=num_latent,
                            is_training=is_training)

def make_decoder(latent_tensor,
                    output_shape,
                    decoder_fn,
                    is_training=True,):
    '''Args:
        output_shape: Tuple with the output shape of the observations to be generated.
        decoder_fn: Function that that takes the arguments (input_tensor, output_shape, is_training) and returns the decoded observations.
        '''
    with tf.variable_scope("decoder"):
        return decoder_fn(latent_tensor=latent_tensor,
                            output_shape=output_shape,
                            is_training=is_training)

def make_discriminator(input_tensor,
                            discriminator_fn,
                            is_training=False,):
    
    with tf.variable_scope("discriminator"):
        logits, probs = discriminator_fn(input_tensor, is_training=is_training)
        clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
    return logits, clipped
    
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

def fc_encoder(input, num_latent):
    '''fully connected encoder'''
    with tf.variable_scope('fc_encoder', reuse=reuse) as scope:
        
        n_layers = {
                    'n_layer_1': 512,
                    'n_layer_2': 512,
        }
        
        output = input
        output = LeakyReLULayer('encoder.1', input.get_shape().as_list()[1], n_layers['n_layer_1'], output)    
        output = LeakyReLULayer('encoder.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
        mu = linear.Linear('encoder.out.mu', n_layers['n_layer_2'], num_latent, output, initialization='he')
        log_var = linear.Linear('encoder.out.var', n_layers['n_layer_2'], num_latent, output, initialization='he')

    return means, log_var

def fc_decoder(input, out_dim):
    '''fully connected decoder'''
    with tf.variable_scope('fc_decoder', reuse=reuse) as scope:
    
        n_layers = {
                    'n_layer_1': 512,
                    'n_layer_2': 512,
        }
        
        output = input
        
        output = LeakyReLULayer('decoder.1', input.get_shape().as_list()[1], n_layers['n_layer_1'], output)   
        output = LeakyReLULayer('decoder.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
        output = LeakyReLULayer('decoder.3', n_layers['n_layer_2'], out_dim, output)
        
    return output

@gin.configurable("fc_discriminator", whitelist=[])
def fc_discriminator(input_tensor, is_training=True):
  """Fully connected discriminator used in FactorVAE paper for all datasets.

  Based on Appendix A page 11 "Disentangling by Factorizing"
  (https://arxiv.org/pdf/1802.05983.pdf)

  Args:
    input_tensor: Input tensor of shape (None, num_latents) to build
      discriminator on.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    logits: Output tensor of shape (batch_size, 2) with logits from
      discriminator.
    probs: Output tensor of shape (batch_size, 2) with probabilities from
      discriminator.
  """
  del is_training
  flattened = tf.layers.flatten(input_tensor)
  d1 = tf.layers.dense(flattened, 1000, activation=tf.nn.leaky_relu, name="d1")
  d2 = tf.layers.dense(d1, 1000, activation=tf.nn.leaky_relu, name="d2")
  d3 = tf.layers.dense(d2, 1000, activation=tf.nn.leaky_relu, name="d3")
  d4 = tf.layers.dense(d3, 1000, activation=tf.nn.leaky_relu, name="d4")
  d5 = tf.layers.dense(d4, 1000, activation=tf.nn.leaky_relu, name="d5")
  d6 = tf.layers.dense(d5, 1000, activation=tf.nn.leaky_relu, name="d6")
  logits = tf.layers.dense(d6, 2, activation=None, name="logits")
  probs = tf.nn.softmax(logits)
  return logits, probs


@gin.configurable("test_encoder", whitelist=["num_latent"])
def test_encoder(input_tensor, num_latent, is_training):
  """Simple encoder for testing.

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training
  flattened = tf.layers.flatten(input_tensor)
  means = tf.layers.dense(flattened, num_latent, activation=None, name="e1")
  log_var = tf.layers.dense(flattened, num_latent, activation=None, name="e2")
  return means, log_var


@gin.configurable("test_decoder", whitelist=[])
def test_decoder(latent_tensor, output_shape, is_training=False):
  """Simple decoder for testing.

  Args:
    latent_tensor: Input tensor to connect decoder to.
    output_shape: Output shape.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
      pixel intensities.
  """
  del is_training
  output = tf.layers.dense(latent_tensor, np.prod(output_shape), name="d1")
  return tf.reshape(output, shape=[-1] + output_shape)
