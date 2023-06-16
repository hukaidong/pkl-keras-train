import tensorflow as tf


def fn_out_of_boundary(_, y_pred):
    abs = tf.abs(y_pred)
    abs_oob = tf.maximum(abs, 1)
    return tf.reduce_mean(abs_oob)


def fn_rmse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)

    loss = tf.reduce_mean(squared_difference, axis=[0, 1])
    return loss


def fn_absmean(y_true, y_pred):
    abs = tf.abs(y_true - y_pred)

    loss = tf.reduce_mean(abs, axis=[0, 1])
    return loss


def fn_loss(y_true, y_pred):
    return fn_rmse(y_true, y_pred) + fn_out_of_boundary(y_true, y_pred)


loss_dict = {
    'fn_loss': fn_loss,
    'fn_rmse': fn_rmse,
    'fn_absmean': fn_absmean,
    'fn_oob': fn_out_of_boundary
}
