import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_custom.sampling import SamplingLayer


class EncoderModel(keras.Model):
    def get_config(self):
        pass

    def __init__(self, hp, *, input_shape, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.dense_units = dense_units = hp.Choice('sampling_units', values=[64, 128, 256])

        self.mean = keras.Sequential()
        self.mean.add(keras.Input(shape=input_shape))
        self.mean.add(layers.Dense(dense_units, name='z_mean'))

        self.log_var = keras.Sequential()
        self.log_var.add(keras.Input(shape=input_shape))
        self.log_var = layers.Dense(dense_units, name='z_log_var')

        self.sampler = SamplingLayer(hp, name='z_prior')


    def get_output_shape(self):
        return self.dense_units

    def call(self, inputs, **kwargs):
        z_mean = self.mean(inputs)
        z_log_var = self.log_var(inputs)
        z = self.sampler([z_mean, z_log_var])
        return z
