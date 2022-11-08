import keras_custom.prelude
import keras_tuner
from load_data import load_dataset
from model_builder import custom_builder, set_reg_fragment

from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from keras_tuner import Hyperband, Objective


class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        set_reg_fragment((epoch+1) / 300.0)


if __name__ == "__main__":
    target = "train.pkl"
    target_val = "val.pkl"
    ns = {"nframe": 8, "nagent": 30, "nhead": 8}
    train_dataset = load_dataset(target, batch_size=1024, **ns)
    val_dataset = load_dataset(target_val, batch_size=1024, **ns)
    data = train_dataset.prepare()
    val_data = val_dataset.prepare()

    model_builder = custom_builder(**ns)

    hp = keras_tuner.HyperParameters()
    hp.values['units-1'] = 1024
    hp.values['units-2'] = 1024
    hp.values['units-3'] = 512
    hp.values['l2-reg'] = 0.01
    hp.values['num-layers'] = 6
    hp.values['learning_rate'] = 1e-04
    hp.values['zerod_alpha'] = 1

    model = model_builder(hp)
    l2_schedule_callback = CustomCallback()
    model.fit(data, validation_data=val_data, epochs=300, verbose=1, callbacks=[l2_schedule_callback])
