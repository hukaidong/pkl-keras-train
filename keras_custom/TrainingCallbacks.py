import tensorflow as tf


def save_best_callback():
    return tf.keras.callbacks.ModelCheckpoint(filepath="best_valid",
                                              save_weights_only=True,
                                              monitor='val_fn_rmse',
                                              mode='min',
                                              save_best_only=True)


def early_stopping_callback():
    return tf.keras.callbacks.EarlyStopping(patience=10,
                                            verbose=1,
                                            monitor='val_fn_rmse',
                                            mode='min')
