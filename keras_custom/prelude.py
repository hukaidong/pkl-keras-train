import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

sys.path.append(os.path.dirname(__file__))

import tensorflow

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

except RuntimeError as e:
    print(e)


import keras_custom.globals
keras_custom.globals.init()

def init_module_cfg():
    with open('model.cfg', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            key, value = line.split('=')
            keras_custom.globals.MODULE_CFG[key.strip()] = eval(value.strip())


def unused():
    pass

