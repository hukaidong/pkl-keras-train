import os
import sys
import argparse
import keras_custom.prelude
from load_data import load_dataset
from model_builder import custom_builder
from keras_hyper import load_parameter_from_tuner, manual_set_parameter
from tensorflow.keras import callbacks


def save_best_callback():
    return callbacks.ModelCheckpoint(filepath="best_valid",
                                     save_weights_only=True,
                                     monitor='val_fn_rmse',
                                     mode='min',
                                     save_best_only=True)


def early_stopping_callback():
    return callbacks.EarlyStopping(patience=10,
                                   verbose=1,
                                   monitor='val_fn_rmse',
                                   mode='min')

parser = argparse.ArgumentParser(
                prog='ProgramName',
                description='What the program does',
                epilog='Text at the bottom of help')

parser.add_argument('filename')           # positional argument
parser.add_argument('--fast', action='store_true')      # option that takes a value
args = parser.parse_args()

if __name__ == "__main__":
    target = 'train.json'
    target_val = 'val.json'
    ns = {"nframe": 14, "nagent": 2, "nhead": 8}

    os.chdir(sys.argv[1])

    model_builder = custom_builder(**ns)
    hp = manual_set_parameter(model_builder)
    model = model_builder(hp)
    len_out = {}
    train_dataset = load_dataset(target,
                                 batch_size=1024,
                                 len_out=len_out,
                                 **ns)
    val_dataset = load_dataset(target_val, batch_size=1024, **ns)
    data = train_dataset.prepare()
    val_data = val_dataset.prepare()

    data_mul = 5000 / len_out['len_out']
    print('loaded train amount:', len_out['len_out'])

    if os.path.exists('result_model.index'):
        print('restore trained model')
        model.load_weights('result_model')

    if not args.fast:  # fast is for debug and run no training
        h = model.fit(data,
                      validation_data=val_data,
                      epochs=50,
                      verbose=2,
                      )

        h = model.fit(data,
                      validation_data=val_data,
                      epochs=int(500 * data_mul),
                      verbose=2,
                      callbacks=[early_stopping_callback()])
    model.save_weights("result_model")

    print(sys.argv[1], "training complete")
