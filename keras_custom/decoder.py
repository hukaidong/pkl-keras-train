import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DecoderModel(keras.Model):
    def get_config(self):
        pass

    def __init__(self, hp, input_shape):
        super(DecoderModel, self).__init__()
        self.nlp = None
        hp1 = hp.Choice('units_1', values=[512, 1024])
        hp2 = hp.Choice('units_2', values=[512, 1024])
        hp3 = hp.Choice('units_3', values=[512, 1024])
        reg_base = hp.Choice('l2_reg', values=[1e-2, 1e-4, 0.0])
        num_layers = hp.Choice('num_layers', values=[3, 4, 5, 6])
        self.latent_shape = input_shape
        self.parameter_size = 9
        self.nlp_build(hp1, hp2, hp3, reg_base, num_layers)

    def get_output_shape(self):
        return self.parameter_size

    def nlp_build(self, hp1, hp2, hp3, reg, num_layers):
        initializer = tf.keras.initializers.GlorotNormal()
        regularizer = tf.keras.regularizers.L2()
        regularizer.l2 = reg
        activation = 'relu'
        dense_opts = {
            'activation': activation,
            'kernel_initializer': initializer,
            'kernel_regularizer': regularizer
        }

        model = keras.Sequential()
        model.add(layers.Input(shape=self.latent_shape))
        model.add(layers.Dense(hp1, **dense_opts))
        if num_layers >= 6: model.add(layers.Dense(hp1, **dense_opts))
        model.add(layers.Dense(hp2, **dense_opts))
        if num_layers >= 5: model.add(layers.Dense(hp2, **dense_opts))
        model.add(layers.Dense(hp3, **dense_opts))
        if num_layers >= 4: model.add(layers.Dense(hp3, **dense_opts))
        model.add(layers.Dense(self.parameter_size))
        self.nlp = model

    def call(self, inputs, **kwargs):
        return self.nlp(inputs, **kwargs)
