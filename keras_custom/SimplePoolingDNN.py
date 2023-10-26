import tensorflow
from tensorflow import keras
from keras_custom.globals import MODULE_CFG
from PredictionLosses import loss_dict as losses


class SimplePoolingDnnModel(keras.Model):
    def __init__(self):
        super(SimplePoolingDnnModel, self).__init__()

        self.nlp_build((MODULE_CFG['nelem'],), MODULE_CFG['parameter_size'])
        input_data = keras.Input(shape=(None, MODULE_CFG['nelem']), ragged=True)
        avg_input = tensorflow.reduce_mean(input_data, axis=1)
        output = self.nlp(avg_input)
        self.model = keras.Model(inputs=input_data, outputs=output)


    def nlp_build(self, input_shape, output_size):
        initializer = keras.initializers.GlorotNormal()
        regularizer = keras.regularizers.l2(0.0001)
        activation = keras.activations.relu
        dense_options = {
            'kernel_initializer': initializer,
            'kernel_regularizer': regularizer,
            'activation': activation
        }


        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        model.add(keras.layers.Dense(256, **dense_options))
        model.add(keras.layers.Dense(256, **dense_options))
        model.add(keras.layers.Dense(output_size))
        self.nlp = model

    def call(self, inputs):
        return self.model(inputs)


def custom_builder():
    input_data = keras.Input(shape=(None, MODULE_CFG['nelem']), ragged=True)
    predictor = SimplePoolingDnnModel()
    outputs = predictor(input_data)
    
    model = keras.models.Model(inputs=[input_data], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                  loss=losses['fn_loss'], metrics=[losses[x] for x in ('fn_rmse', 'fn_absmean')])
    return model
