import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

sys.path.append(os.path.dirname(__file__))

import tensorflow

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
try:
    print("Num GPUs Available: ", len(gpus), file=sys.stderr)
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

except RuntimeError as e:
    print(e)


from yaml import load, Loader
import keras_custom.globals
keras_custom.globals.init()

def init_module_cfg():
    with open('model.yml', 'r') as f:
        keras_custom.globals.MODULE_CFG.update(load(f, Loader=Loader))


def unused():
    pass

