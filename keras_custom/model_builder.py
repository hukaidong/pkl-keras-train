import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

_srf = []
def set_reg_fragment(value):
    _srf[0](value)

def custom_builder(nframe=18, nagent=30, nhead=8, is_natural=True, debug_dict=None):
    
    def custom_loss(beta):
        def fn_loss(y_true, y_pred):
            expand_dim = tf.expand_dims(y_true, axis=2)
            y_true = tf.tile(expand_dim, (1, 1, nhead))
            squared_difference = tf.square(y_true - y_pred)
            if is_natural:
                zerod_difference = 1 - tf.math.abs(y_pred-0.5)
            else:
                zerod_difference = 1 - tf.math.abs(y_pred)

            loss = tf.reduce_mean(squared_difference, axis=[0, 1]) + beta * tf.reduce_mean(zerod_difference, axis=[0, 1])
            return loss
        return fn_loss

    def custom_metric(y_true, y_pred):
        expand_dim = tf.expand_dims(y_true, axis=2)
        y_true = tf.tile(expand_dim, (1, 1, nhead))
        y_pred_int = arr_as_integer(y_pred)
        y_true_int = arr_as_integer(y_true)
        y_equals = tf.cast(tf.math.equal(y_pred_int, y_true_int), tf.float32)
        accuracy = tf.reduce_mean(y_equals, axis=[1, 2], name="accuracy")
        return accuracy

    def custom_metric_ensemble(y_true, y_pred):
        y_true_int = arr_as_integer(y_true)
        y_pred_int = arr_as_integer(y_pred)
        round_up_head = (nhead - 1) // 2 * 2 + 1
        ensemble_idx = tf.gather(tf.random.shuffle(tf.range(nhead)), tf.range(round_up_head))
        y_pred_sel = tf.cast(tf.gather(y_pred_int, ensemble_idx, axis=2), tf.float32)
        y_pred_mean = tf.reduce_mean(y_pred_sel, axis=[2])
        y_vote = arr_as_integer(y_pred_mean)
        y_equals = tf.cast(tf.math.equal(y_vote, y_true_int), tf.float32)
        accuracy = tf.reduce_mean(y_equals, axis=[1], name="ensemble_accuracy")
        return accuracy

    if is_natural:
        # array have binary value {0, 1}
        def arr_as_integer(arr):
            return tf.cast(tf.greater_equal(arr, 0.5), tf.int32)
    else:
        # array have binary value {-1, 1}
        def arr_as_integer(arr):
            return tf.cast(tf.greater_equal(arr, 0), tf.int32) * 2 - 1


    def data_augmentation():
        input_shape = (nframe * nagent, nhead)
        sep_shape = (nframe, nagent, nhead)

        input_img = keras.Input(shape=input_shape)
        sep_img = layers.Reshape(sep_shape)(input_img)
        # Get random shuffle order
        order = layers.Lambda(lambda x: tf.random.shuffle(tf.range(x)))(nagent)
        # Apply shuffle
        tensor = layers.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], tf.int32), axis=2, ))([sep_img, order])
        # Merge frame_dim and agent_dim to get an input_shape tensor
        # sep_tensor = layers.Reshape(input_shape)(tensor)

        return keras.models.Model(
            inputs=[input_img],
            outputs=tensor,
        )

    def model_builder_colon(hp1, hp2, hp3, reg, num_layers):
        model = keras.Sequential()
        initializer = tf.keras.initializers.GlorotNormal()
        regularizer = tf.keras.regularizers.L2()
        regularizer.l2 = reg
        activation = 'relu'

        model.add(layers.Dense(hp1, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        if num_layers >= 6:
            model.add(layers.Dense(hp1, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Dense(hp2, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        if num_layers >= 5:
            model.add(layers.Dense(hp2, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Dense(hp3, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))
        if num_layers >= 4:
            model.add(layers.Dense(hp3, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer))

        model.add(layers.Dense(8))
        return model

    def model_builder(hp):
        time_agent_vec_shape = (nframe, nagent, nhead)
        input_shape = (nframe * nagent, nhead)
        input_img = keras.Input(shape=input_shape)
        agu_img = data_augmentation()(input_img)
        col_out = []
        hp1 = hp.Choice('units-1', values=[512, 1024])
        hp2 = hp.Choice('units-2', values=[512, 1024])
        hp3 = hp.Choice('units-3', values=[512, 1024])
        reg_base = hp.Choice('l2-reg', values=[1e-2, 1e-4, 0.0])
        num_layers = hp.Choice('num-layers', values=[3, 4, 5, 6])
        reg = K.variable(0.0, name='regularize-epoch')

        def _set_reg_fragment(frag):
            K.set_value(reg, reg_base * frag)
        _srf.append(_set_reg_fragment)

        for i in range(nhead):
            col_img = tf.gather(agu_img, i, axis=3)
            col_flat = layers.Flatten()(col_img)
            col_out.append(model_builder_colon(hp1, hp2, hp3, reg, num_layers)(col_flat))

        outputs = tf.stack(col_out, axis=2)

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-5])
        hp_alpha = hp.Choice('zerod_alpha', values=[1.0, 1e-3])
        model = keras.models.Model(inputs=[input_img], outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=custom_loss(hp_alpha), metrics=[custom_metric, custom_metric_ensemble])
        return model

    if debug_dict is not None:
        debug_dict.update(locals())
        del debug_dict['debug_dict']


    return model_builder
