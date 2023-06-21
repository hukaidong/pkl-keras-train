import os
import sys
import numpy as np
import argparse

from keras_custom.prelude import init_module_cfg
from keras_custom.globals import MODULE_CFG, HYPER_PARAMS
from keras_tuner import HyperParameters
from keras_hyper import update_hp_parameters
from SegmentedTrajectories import SegmentedTrajectories
from SegmentedAutoencoder import custom_builder, register_hyper_params

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('latent_dir')
    parser.add_argument(
        '-c', '--count',
        default=1000, type=int
    )
    parser.add_argument(
        '--parsable',
        default=False,
        action='store_true'
    )
    return parser


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    if not os.getenv('KERAS_DEVELOP'):
        os.chdir(args.latent_dir)

    init_module_cfg()

    target = 'train.json'
    target_val = 'val.json'

    train_dataset = SegmentedTrajectories(target)
    val_dataset = SegmentedTrajectories(target_val)
    data = train_dataset.prepare(batch_size=256)
    val_data = val_dataset.prepare(batch_size=256)

    autoencoder_frag = {}
    register_hyper_params()
    update_hp_parameters(HYPER_PARAMS)

    model = custom_builder(HYPER_PARAMS, autoencoder_fragment_out=autoencoder_frag)
    encoder, decoder = autoencoder_frag['autoencoder_fragment_out']
    model.load_weights("result_model").expect_partial()

    random_latent = np.random.normal(size=(args.count, HYPER_PARAMS.values['sampling_units']))
    sample_param = decoder(random_latent).numpy()
    sample_param = (sample_param+1)/2

    for row in sample_param:
        if args.parsable:
            print(*["{0:.6f}".format(i) for i in row], sep=',')
        else:
            print(*["{0: .6f}".format(i) for i in row], sep=', ')
