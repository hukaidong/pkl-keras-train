import keras_custom.prelude
import keras_tuner
from model_builder import custom_builder


if __name__ == "__main__":
    ns = {"nframe": 14, "nagent": 2, "nhead": 8}
    model_builder = custom_builder(**ns)
    tuner = keras_tuner.Hyperband(model_builder, objective="val_loss")
    tuner.results_summary(10)
