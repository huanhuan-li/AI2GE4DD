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
        output = linear.Linear('decoder.3', n_layers['n_layer_2'], out_dim, output, initialization='he')
        
    return output

def fc_discriminator(input, out_dim):

    with tf.variable_scope('fc_discriminator', reuse=reuse) as scope:
    
        n_layers = {
                    'n_layer_1': 512,
                    'n_layer_2': 512,
        }
        
        output = input
        
        output = LeakyReLULayer('disc.1', input.get_shape().as_list()[1], n_layers['n_layer_1'], output)
        output = LeakyReLULayer('disc.2', n_layers['n_layer_1'], n_layers['n_layer_2'], output)
        logits = linear.Linear('disc.logits', n_layers['n_layer_2'], out_dim, output, initialization='he')
        probs = tf.nn.softmax(logits)
    return logits, probs