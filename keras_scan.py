import keras_custom.prelude
import keras_tuner
from load_data import load_dataset
from model_builder import custom_builder, set_reg_fragment

from tensorflow.keras import callbacks

class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        set_reg_fragment((epoch + 1) / 300.0)


if __name__ == "__main__":
    target = "train.pkl"
    target_val = "val.pkl"
    ns = {"nframe": 14, "nagent": 2, "nhead": 8}
    train_dataset = load_dataset(target, batch_size=1024, **ns)
    val_dataset = load_dataset(target_val, batch_size=1024, **ns)
    data = train_dataset.prepare()
    val_data = val_dataset.prepare()

    model_builder = custom_builder(**ns)

    tuner = keras_tuner.Hyperband(model_builder, objective="val_loss", max_epochs=500, hyperband_iterations=3)
    sample_model = model_builder(keras_tuner.HyperParameters())
    print(sample_model.summary())
    l2_schedule_callback = CustomCallback()
    tuner.search(data, validation_data=val_data, epochs=300, verbose=1, callbacks=[l2_schedule_callback])
