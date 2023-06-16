import os
import sys
from keras_custom.prelude import init_module_cfg
from keras_custom.globals import MODULE_CFG, HYPER_PARAMS
import keras_tuner
from keras_hyper import update_hp_parameters
from TrainingCallbacks import save_best_callback, early_stopping_callback
from SegmentedTrajectories import SegmentedTrajectories
from SegmentedAutoencoder import custom_builder, register_hyper_params

if __name__ == "__main__":
    if not os.getenv('KERAS_DEVELOP'):
        os.chdir(sys.argv[1])

    init_module_cfg()

    target = 'train.json'
    target_val = 'val.json'

    train_dataset = SegmentedTrajectories(target)
    val_dataset = SegmentedTrajectories(target_val)
    data = train_dataset.prepare(batch_size=256)
    val_data = val_dataset.prepare(batch_size=256)

    HYPER_PARAMS = keras_tuner.HyperParameters()
    register_hyper_params()
    update_hp_parameters(HYPER_PARAMS)

    model = custom_builder()

    if False and os.path.exists('result_model.index'):
        print('restore trained model')
        model.load_weights('result_model')

    epochs = int(MODULE_CFG.get('epochs', 500))

    h = model.fit(data,
                  validation_data=val_data,
                  epochs=epochs,
                  verbose=(1 if sys.stdout.isatty() else 2),
                  #callbacks=[early_stopping_callback()],
                  )
    model.save_weights("result_model")

    if not os.getenv('KERAS_DEVELOP'):
        print(sys.argv[1], "training complete")
