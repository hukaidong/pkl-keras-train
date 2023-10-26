import os
import sys

from keras_custom.prelude import init_module_cfg
from globals import MODULE_CFG

from PredictionLosses import loss_dict as losses
from SegmentedTrajectories import SegmentedTrajectories
from TrainingCallbacks import early_stopping_callback
from SimplePoolingDNN import custom_builder

if __name__ == '__main__':
    os.chdir(sys.argv[1])
    init_module_cfg()

    train_dataset = SegmentedTrajectories("train.json")
    val_dataset = SegmentedTrajectories("val.json")
    data = train_dataset.prepare(batch_size=1024)
    val_data = val_dataset.prepare(batch_size=1024)

    model = custom_builder()

    h = model.fit(data,
                  validation_data=val_data,
                  epochs=500,
                  verbose=(1 if sys.stdout.isatty() else 2),
                  callbacks=[early_stopping_callback()],
                  )

    model.save_weights("plain_result_model")
