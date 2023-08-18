from keras_custom.prelude import init_module_cfg
from keras_custom.globals import MODULE_CFG, HYPER_PARAMS

from SegmentedTrajectories import SegmentedTrajectories
from SegmentedAutoencoder import hp_tuning_builder, custom_builder

import keras_tuner
import os
import sys


if __name__ == "__main__":
    if not os.getenv('KERAS_DEVELOP'):
        os.chdir(sys.argv[1])

    init_module_cfg()

    target = 'train.json'
    target_val = 'val.json'
    train_dataset = SegmentedTrajectories(target)
    val_dataset = SegmentedTrajectories(target_val)
    data = train_dataset.prepare(batch_size=1024)
    val_data = val_dataset.prepare(batch_size=1024)

    model_builder = custom_builder
    obj = keras_tuner.Objective("val_fn_rmse", direction='min')
    tuner = keras_tuner.Hyperband(model_builder, objective=obj, max_epochs=500, hyperband_iterations=3)
    sample_model = model_builder(keras_tuner.HyperParameters())
    print(sample_model.summary())
    tuner.search(data, validation_data=val_data, epochs=300, verbose=1)
    print(tuner.results_summary())
