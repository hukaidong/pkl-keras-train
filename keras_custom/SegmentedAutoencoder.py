import tensorflow as tf

from keras_custom.globals import MODULE_CFG, HYPER_PARAMS, set_hyper_params
from keras_custom.encoder import EncoderModel
from keras_custom.decoder import DecoderModel
from keras_custom.PredictionLosses import loss_dict as losses
from keras_custom.SegmentedPoolingEncoder import SegmentedPoolingEncoderModel


def hp_tuning_builder(hp_):
    set_hyper_params(hp_)

    return custom_builder(HYPER_PARAMS)

def register_hyper_params():
    custom_builder(HYPER_PARAMS)


# custom_builder is the function that will be called by keras tuner
# it requires a global HYPER_PARAMS to be defined, usually being used
# after hyper-parameter tuning. the best hyper-parameters can be pre-
# defined from file and be loaded ahead. Otherwise, tuning the model
# with hp_tuning_builder
def custom_builder(hp, autoencoder_fragment_out=None):
    input_data = tf.keras.Input(shape=(None, MODULE_CFG['nelem']), ragged=True)
    encoder = SegmentedPoolingEncoderModel(hp)
    decoder = DecoderModel(hp, input_shape=encoder.get_output_shape())
    latent = encoder(input_data)
    outputs = decoder(latent)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-5, 2e-6])

    model = tf.keras.models.Model(inputs=[input_data], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=losses['fn_loss'], metrics=[losses[x] for x in ('fn_rmse', 'fn_absmean')])

    if autoencoder_fragment_out is not None:
        autoencoder_fragment_out['autoencoder_fragment_out'] = [encoder, decoder]

    return model
