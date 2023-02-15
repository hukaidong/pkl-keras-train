import tensorflow as tf
from tensorflow.keras import layers


class SamplingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, hp, **kwargs):
        super(SamplingLayer, self).__init__(**kwargs)
        prior_weight = hp.Choice('prior_weight', values=[1.0, 0.3, 0.1])
        self.loss_weight = prior_weight

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        self.add_loss(self.loss_weight * tf.reduce_mean(tf.square(z)))
        return z
