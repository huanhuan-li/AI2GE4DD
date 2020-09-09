import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BatchNorm(layers.Layer):
    def __init__(self, is_training=False):
        super(BatchNorm, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                     momentum=0.9,
                                                     scale=True,
                                                     trainable=is_training)

    def call(self, inputs, training):
        x = self.bn(inputs, training=training)
        return x
        
class DenseLayer(layers.Layer):
    def __init__(self, hidden_n, is_input=False):
        super(DenseLayer, self).__init__()

        self.fc_op = layers.Dense(hidden_n,
                                  kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                  bias_initializer=keras.initializers.Constant(value=0.0))

    def call(self, inputs):
        x = self.fc_op(inputs)

        return x
        
class Sigmoid(layers.Layer):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def call(self, inputs):
        return keras.activations.sigmoid(inputs)


class Tanh(layers.Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def call(self, inputs):
        return keras.activations.tanh(inputs)