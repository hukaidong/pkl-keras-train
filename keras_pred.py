import sys
import numpy as np
import keras_custom.prelude
import keras_tuner
from load_data import load_dataset
from model_builder import custom_builder, set_reg_fragment

from tensorflow.keras import callbacks

class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # set_reg_fragment((epoch + 1) / 300.0)
        set_reg_fragment(1.0)


if __name__ == "__main__":
    target = sys.argv[1]
    target_val = sys.argv[2]
    ns = {"nframe": 14, "nagent": 2, "nhead": 8}
    train_dataset = load_dataset(target, batch_size=1024, **ns)
    val_dataset = load_dataset(target_val, batch_size=1024, **ns)
    data = train_dataset.prepare()
    val_data = val_dataset.prepare()

    model_builder = custom_builder(**ns)
    #tuner = keras_tuner.Hyperband(model_builder, objective="val_loss")
    l2_schedule_callback = CustomCallback()
    #hp = tuner.get_best_hyperparameters()[0]
    hp = keras_tuner.HyperParameters()
    hp.values['sampling_units'] = 256
    hp.values['unit-1'] = 512
    hp.values['unit-2'] = 512
    hp.values['unit-3'] = 512
    hp.values['l2_reg'] = 0.001
    hp.values['prior-weight'] = 0.01
    hp.values['num-layers'] = 5
    hp.values['learning_rate'] = 2e-6
    encoder = None
    decoder = None
    def set_decoder(models):
        global encoder, decoder
        encoder, decoder = models

    model = model_builder(hp, decoder_out=set_decoder)
    model.load_weights("./best_valid")
    print(model.evaluate(val_data))


    #predict = model.predict(val_data)
    #ground = [x[1][0, :] for x in val_data.as_numpy_iterator()]
    #np.savetxt("gt.npy", np.asarray(ground))
    #np.savetxt("pd.npy", predict)
