import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_custom.encoder import EncoderModel
from keras_custom.decoder import DecoderModel


def custom_builder(nframe=18, nagent=30, nhead=8, is_natural=False, debug_dict=None):
    def custom_loss():
        def fn_loss(y_true, y_pred):
            squared_difference = tf.square(y_true - y_pred)

            loss = tf.reduce_mean(squared_difference, axis=[0, 1])
            return loss

        def fn_error(y_true, y_pred):
            abs = tf.abs(y_true - y_pred)

            loss = tf.reduce_mean(abs, axis=[0, 1])
            return loss

        return fn_loss, fn_error

    # def flatten_augmentation():
    #     input_arr = keras.Input((nframe*nagent, nhead))
    #     input_rshp = layers.Reshape((nframe, nagent, nhead))(input_arr)
    #     input_avg = tf.reduce_mean(input_rshp, axis=1)
    #     input_fltn = layers.Reshape((nagent*nhead,))(input_avg)
    #     return keras.models.Model(inputs=[input_arr], outputs=input_fltn)

    def flatten_augmentation():
        return layers.Flatten()

    def model_builder(hp, decoder_out=None):
        input_shape = (nframe * nagent, nhead)

        input_img = keras.Input(shape=input_shape)
        agu_img = flatten_augmentation()(input_img)

        encoder = EncoderModel(hp, input_shape=agu_img.shape[1])
        decoder = DecoderModel(hp, input_shape=encoder.get_output_shape())

        outputs = decoder(encoder(agu_img))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-5, 2e-6])
        model = keras.models.Model(inputs=[input_img], outputs=outputs)
        loss, error = custom_loss()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=loss, metrics=[loss, error])
        if decoder_out:
            decoder_out([encoder, decoder])
        return model

    if debug_dict is not None:
        debug_dict.update(locals())
        del debug_dict['debug_dict']

    return model_builder
