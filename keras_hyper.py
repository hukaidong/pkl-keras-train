import keras_tuner

preset_parameters_keys = ('sampling_units,prior_weight,units_1,units_2,units_3,l2_reg,num_layers,'
                          'learning_rate').split(',')
preset_parameters_vals = '256,1.0,512,512,512,0.001,5,1.0e-05'.split(',')
preset_parameters_dict = {key: eval(val) for key, val in zip(preset_parameters_keys, preset_parameters_vals)}


def load_parameter_from_tuner(model_builder):
    obj = keras_tuner.Objective("val_fn_loss", direction='min')
    tuner = keras_tuner.Hyperband(model_builder, objective=obj)
    hp = tuner.get_best_hyperparameters()[0]
    return hp


def manual_set_parameter(model_builder):
    global preset_parameters_dict, preset_parameters_keys, preset_parameters_vals

    hp = keras_tuner.HyperParameters()
    model_builder(hp)  # register all hyper-parameters
    return update_hp_parameters(hp)


def update_hp_parameters(hp):
    global preset_parameters_dict, preset_parameters_keys, preset_parameters_vals

    hp.values.update(preset_parameters_dict)
    return hp
