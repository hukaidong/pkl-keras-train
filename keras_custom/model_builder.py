import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

_srf = []


def set_reg_fragment(value):
    _srf[0](value)


def custom_builder(nframe=18, nagent=30, nhead=8, is_natural=False, debug_dict=None):
    def custom_loss():
        def fn_loss(y_true, y_pred):
            squared_difference = tf.square(y_true - y_pred)

            loss = tf.reduce_mean(squared_difference, axis=[0, 1])
            return loss

        return fn_loss

    # def flatten_augmentation():
    #     input_arr = keras.Input((nframe*nagent, nhead))
    #     input_rshp = layers.Reshape((nframe, nagent, nhead))(input_arr)
    #     input_avg = tf.reduce_mean(input_rshp, axis=1)
    #     input_fltn = layers.Reshape((nagent*nhead,))(input_avg)
    #     return keras.models.Model(inputs=[input_arr], outputs=input_fltn)

    def flatten_augmentation():
        return layers.Flatten()

    def model_builder_colon(hp1, hp2, hp3, reg, num_layers):
        model = keras.Sequential()
        initializer = tf.keras.initializers.GlorotNormal()
        regularizer = tf.keras.regularizers.L2()
        regularizer.l2 = reg
        activation = 'relu'

        model.add(
            layers.Dense(hp1, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        if num_layers >= 6:
            model.add(layers.Dense(hp1, activation=activation, kernel_initializer=initializer,
                                   kernel_regularizer=regularizer))
        model.add(
            layers.Dense(hp2, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        if num_layers >= 5:
            model.add(layers.Dense(hp2, activation=activation, kernel_initializer=initializer,
                                   kernel_regularizer=regularizer))
        model.add(
            layers.Dense(hp3, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        if num_layers >= 4:
            model.add(layers.Dense(hp3, activation=activation, kernel_initializer=initializer,
                                   kernel_regularizer=regularizer))

        model.add(layers.Dense(9))
        return model

    def model_builder(hp):
        input_shape = (nframe * nagent, nhead)
        input_img = keras.Input(shape=input_shape)
        agu_img = flatten_augmentation()(input_img)

        hp1 = hp.Choice('units-1', values=[512, 1024])
        hp2 = hp.Choice('units-2', values=[512, 1024])
        hp3 = hp.Choice('units-3', values=[512, 1024])
        reg_base = hp.Choice('l2-reg', values=[1e-2, 1e-4, 0.0])
        num_layers = hp.Choice('num-layers', values=[3, 4, 5, 6])
        reg = K.variable(0.0, name='regularize-epoch')

        def _set_reg_fragment(frag):
            K.set_value(reg, reg_base * frag)

        _srf.append(_set_reg_fragment)

        outputs = model_builder_colon(hp1, hp2, hp3, reg, num_layers)(agu_img)

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-5, 2e-6])
        model = keras.models.Model(inputs=[input_img], outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=custom_loss())
        return model

    if debug_dict is not None:
        debug_dict.update(locals())
        del debug_dict['debug_dict']

    return model_builder
