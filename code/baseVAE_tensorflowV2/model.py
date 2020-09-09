from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,metrics
from utils import *


class VAE2DD:
    
    def __init__(self, xdim, ddim, ydim, zdim, lr, beta1, beta2):
        
        self.xdim = xdim
        self.ddim = ddim
        self.ydim = ydim
        self.zdim = zdim
        self.encodeX = self.encodeX_model(is_training=True)
        self.encodeD = self.encodeD_model(is_training=True)
        self.decodeX = self.decodeX_model(is_training=True)
        self.decodeD = self.decodeD_model(is_training=True)
        self.decodeY = self.decodeY_model(is_training=True)
        self.optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=beta1, beta_2=beta2)
        self.x_rec_loss_metric = tf.keras.metrics.Mean('Loss/rec/x_rec_loss', dtype=tf.float32)
        self.y_rec_loss_metric = tf.keras.metrics.Mean('Loss/rec/y_rec_loss', dtype=tf.float32)
        self.d_rec_loss_metric = tf.keras.metrics.Mean('Loss/rec/d_rec_loss', dtype=tf.float32)
        self.x_kl_loss_metric = tf.keras.metrics.Mean('Loss/kl/x_kl_loss', dtype=tf.float32)
        self.d_kl_loss_metric = tf.keras.metrics.Mean('Loss/kl/d_kl_loss', dtype=tf.float32)
        self.total_loss_metric = tf.keras.metrics.Mean('Loss/totol_loss', dtype=tf.float32)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), 
                                                optimizer=self.optimizer,
                                                encodeX=self.encodeX,
                                                encodeD=self.encodeD,
                                                decodeX=self.decodeX,
                                                decodeD=self.decodeD,
                                                decodeY=self.decodeY)
        
    def encodeX_model(self, is_training):
        model = tf.keras.Sequential()
        model.add(DenseLayer(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(256))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(DenseLayer(256))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(DenseLayer(2*self.zdim))
        return model
        
    def encodeD_model(self, is_training):
        model = tf.keras.Sequential()
        model.add(DenseLayer(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(256))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(DenseLayer(256))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(DenseLayer(2*self.zdim))
        return model
        
    def decodeX_model(self, is_training):
        model = tf.keras.Sequential()
        model.add(DenseLayer(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(self.xdim))
        return model
        
    def decodeD_model(self, is_training):
        model = tf.keras.Sequential()
        model.add(DenseLayer(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(self.ddim))
        return model
        
    def decodeY_model(self, is_training):
        model = tf.keras.Sequential()
        model.add(DenseLayer(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(BatchNorm(is_training=is_training))
        model.add(DenseLayer(self.ydim))
        return model
        
    def sample_from_latent_distribution(self, z_mean, z_logvar):
        return tf.add(z_mean, tf.exp(z_logvar / 2) * tf.random.normal(tf.shape(z_mean), 0, 1), name="sampled_latent_variable")

    @tf.function
    def train_one_step(self, batch_x, batch_y, batch_d):
        with tf.GradientTape() as gradient_tape:
            # zx & px
            zx_gaussian_params=self.encodeX(batch_x, training=True)
            zx_mu = zx_gaussian_params[:, :self.zdim]
            zx_logvar = zx_gaussian_params[:, self.zdim:]
            zx_samp = self.sample_from_latent_distribution(zx_mu, zx_logvar)
            px = self.decodeX(zx_samp, training=True)
            # zd & pd
            zd_gaussian_params=self.encodeD(batch_d, training=True)
            zd_mu = zd_gaussian_params[:, :self.zdim]
            zd_logvar = zd_gaussian_params[:, self.zdim:]
            zd_samp = self.sample_from_latent_distribution(zd_mu, zd_logvar)
            pd = self.decodeD(zd_samp, training=True)
            # zy & py
            zy_samp = tf.multiply(zx_samp, zd_samp, name='zd_act_on_zx')
            py = self.decodeY(zy_samp, training=True)
            '''loss'''
            self.x_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(batch_x - px), 1))
            self.d_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_d, logits=pd), 1))
            self.y_rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(batch_y - py), 1))
            self.x_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(zx_mu) + tf.exp(zx_logvar) - zx_logvar - 1, [1]), name="x_kl_loss")
            self.d_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(zd_mu) + tf.exp(zd_logvar) - zd_logvar - 1, [1]), name="d_kl_loss")
            self.loss = self.x_rec_loss + self.d_rec_loss + self.y_rec_loss + self.x_kl_loss + self.d_kl_loss 
            
        self.trainable_variables=self.encodeX.trainable_variables + self.encodeD.trainable_variables + self.decodeX.trainable_variables + self.decodeD.trainable_variables + self.decodeY.trainable_variables
        gradients = gradient_tape.gradient(self.loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.x_rec_loss_metric(self.x_rec_loss)
        self.y_rec_loss_metric(self.y_rec_loss)
        self.d_rec_loss_metric(self.d_rec_loss)
        self.x_kl_loss_metric(self.x_kl_loss)
        self.d_kl_loss_metric(self.d_kl_loss)
        self.total_loss_metric(self.loss)