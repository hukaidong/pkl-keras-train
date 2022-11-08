import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

sys.path.append(os.path.dirname(__file__))

import tensorflow

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        print(gpu)
        tensorflow.config.experimental.set_memory_growth(gpu, True)

except RuntimeError as e:
    print(e)


def unused():
    pass
