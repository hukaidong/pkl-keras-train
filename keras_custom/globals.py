import keras_tuner


def init():
    global MODULE_CFG
    global HYPER_PARAMS
    MODULE_CFG = {}
    HYPER_PARAMS = keras_tuner.HyperParameters()

def set_hyper_params(hp):
    global HYPER_PARAMS
    HYPER_PARAMS = hp