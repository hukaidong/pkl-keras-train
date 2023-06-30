import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_custom.encoder import EncoderModel
from keras_custom.globals import MODULE_CFG


class SegmentedPoolingEncoderModel(keras.Model):
    def get_config(self):
        pass

    def __init__(self, hp, **kwargs):
        super(SegmentedPoolingEncoderModel, self).__init__(**kwargs)
        input_data = tf.keras.Input(shape=(None, MODULE_CFG['nelem']), ragged=True)
        self.encoder = EncoderModel(hp, input_shape=(MODULE_CFG['nelem'],))
        latents = tf.keras.layers.TimeDistributed(self.encoder)(input_data)
        avg_latents = tf.reduce_mean(latents, axis=1)
        self.model = tf.keras.models.Model(inputs=[input_data], outputs=avg_latents)

        if MODULE_CFG.get('segmented', False):
            prior_weight = hp.Choice('prior_weight', values=[1.0, 0.3, 0.1])
            self.loss_weight = prior_weight

    def get_output_shape(self):
        return self.encoder.get_output_shape()

    def call(self, inputs, **kwargs):
        z = self.model(inputs)
        if MODULE_CFG.get('segmented', False):
            self.add_loss(self.loss_weight * tf.reduce_mean(tf.square(z)))
        return z
