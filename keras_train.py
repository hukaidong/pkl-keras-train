import os
import sys
import keras_custom.prelude
import keras_tuner
from load_data import load_dataset
from model_builder import custom_builder

from tensorflow.keras import callbacks

def load_parameter_from_tuner():
    tuner = keras_tuner.Hyperband(model_builder, objective="val_loss")
    hp = tuner.get_best_hyperparameters()[0]
    return hp


if __name__ == "__main__":
    os.chdir(sys.argv[1])
    target = 'train.json'
    target_val = 'val.json'
    ns = {"nframe": 14, "nagent": 2, "nhead": 8}
    train_dataset = load_dataset(target, batch_size=1024, **ns)
    val_dataset = load_dataset(target_val, batch_size=1024, **ns)
    data = train_dataset.prepare()
    val_data = val_dataset.prepare()

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="best_valid",
        save_weights_only=True,
        monitor='val_fn_loss',
        mode='min',
        save_best_only=True)

    model_builder = custom_builder(**ns)

    preset_parameters_keys = ('trial_id,sampling_units,prior_weight,units_1,units_2,units_3,l2_reg,num_layers,'
                              'learning_rate').split(',')
    preset_parameters_vals = '2438,256,1.0,1024,512,512,0.0001,4,2.0e-06'.split(',')
    preset_parameters_dict = {key: eval(val) for key, val in zip(preset_parameters_keys, preset_parameters_vals)}

    hp = keras_tuner.HyperParameters()
    hp.values.update(preset_parameters_dict)

    model = model_builder(hp)
    model.fit(data, validation_data=val_data, epochs=300, verbose=1, callbacks=[model_checkpoint_callback])
    model.save_weights("result_model")
    print(sys.argv[1], "training complete")
