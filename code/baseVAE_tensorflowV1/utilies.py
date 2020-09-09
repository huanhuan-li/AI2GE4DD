# coding=utf-8

import numpy as np
import tensorflow as tf
from scipy.stats import percentileofscore

def compute_gaussian_kl(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]), name="kl_loss")
    
def batchnorm(inputs, axis, scale=None, offset=None, variance_epsilon=0.001, name=None):
    with tf.variable_scope('batchnorm'):
        mean, var = tf.nn.moments(inputs, axis, keep_dims=True)
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, variance_epsilon, name=name)
    return result
    
def anneal(c_max, step, iteration_threshold):
    """Anneal function for anneal_vae https://arxiv.org/abs/1804.03599
    Args:
        c_max: Maximum capacity.
        step: Current step.
        iteration_threshold: How many iterations to reach c_max.
    Returns:
        Capacity annealed linearly until c_max."""
    return tf.math.minimum(c_max * 1., c_max * 1. * tf.to_float(step) / iteration_threshold)
    
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
    
def calcu_rsquare_distance(ref_, samp_):
    result = []
    for gene_idx in range(ref_.shape[1]):
        ref = ref_[:, gene_idx]
        samp = samp_[:, gene_idx]
        
        sample = [0.02] + [i for i in np.arange(0.05, 1.0, 0.05)]
        
        samp_pct_x = []
        samp_pct_y = []
        
        for i,s in enumerate(sample):
            # theoretical quantiles
            samp_pct_x.append(percentileofscore(ref, s))
            # sample quantiles
            samp_pct_y.append(percentileofscore(samp, s))
        
        # estimated quantile distance
        r = np.linalg.norm(np.subtract(np.asarray(samp_pct_x)/100, np.asarray(samp_pct_y)/100), ord=2)
        result.append(r)
    return result
    
class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'
    
class CharVocab:
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string

class OneHotVocab(CharVocab):
    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))