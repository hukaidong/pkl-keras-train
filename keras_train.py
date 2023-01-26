import sys
import keras_custom.prelude
import keras_tuner
from load_data import load_dataset
from model_builder import custom_builder, set_reg_fragment

from tensorflow.keras import callbacks

class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        set_reg_fragment((epoch + 1) / 300.0)

def load_parameter_from_tuner():
    tuner = keras_tuner.Hyperband(model_builder, objective="val_loss")
    hp = tuner.get_best_hyperparameters()[0]
    return hp


if __name__ == "__main__":
    target = sys.argv[1]
    target_val = sys.argv[2]
    ns = {"nframe": 14, "nagent": 2, "nhead": 8}
    train_dataset = load_dataset(target, batch_size=1024, **ns)
    val_dataset = load_dataset(target_val, batch_size=1024, **ns)
    data = train_dataset.prepare()
    val_data = val_dataset.prepare()

    #l2_schedule_callback = CustomCallback()
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="best_valid",
        save_weights_only=True,
        monitor='val_fn_loss',
        mode='min',
        save_best_only=True)

    model_builder = custom_builder(**ns)
    hp = keras_tuner.HyperParameters()
    hp.values['sampling_units'] = 256
    hp.values['unit-1'] = 512
    hp.values['unit-2'] = 512
    hp.values['unit-3'] = 512
    hp.values['l2_reg'] = 0.001
    hp.values['prior-weight'] = 0.01
    hp.values['num-layers'] = 5
    hp.values['learning_rate'] = 2e-6
    model = model_builder(hp)
    model.fit(data, validation_data=val_data, epochs=300, verbose=1, callbacks=[model_checkpoint_callback])
    model.save_weights("result_model")
