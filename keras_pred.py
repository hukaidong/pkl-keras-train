import numpy as np

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
    tuner = keras_tuner.Hyperband(model_builder, objective="val_loss")
    l2_schedule_callback = CustomCallback()
    hp = tuner.get_best_hyperparameters()[0]
    hp.values['learning_rate'] = 3e-6
    model = model_builder(hp)
    model.load_weights("./best_valid")
    predict = model.predict(val_data)
    ground = [x[1][0, :] for x in val_data.as_numpy_iterator()]
    np.savetxt("gt.npy", np.asarray(ground))
    np.savetxt("pd.npy", predict)
