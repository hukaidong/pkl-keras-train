import os
import sys
import keras_custom.prelude
import numpy as np
import argparse
from load_data import load_dataset
from model_builder import custom_builder

from keras_hyper import manual_set_parameter

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
    os.chdir(args.latent_dir)

    config_dict = {}
    with open('model.cfg', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            key, value = line.split('=')
            config_dict[key.strip()] = value.strip()

    target = 'train.json'
    target_val = 'val.json'

    ns = {"nframe": 14, "nagent": int(config_dict['nagent']), "nhead": 8}

    autoencoder_frag = {}
    model_builder = custom_builder(**ns)
    hp = manual_set_parameter(model_builder)
    model = model_builder(hp, autoencoder_fragment_out=autoencoder_frag)
    encoder, decoder = autoencoder_frag['autoencoder_fragment_out']
    model.load_weights("result_model").expect_partial()

    train_dataset = load_dataset(target, batch_size=1024, **ns)
    val_dataset = load_dataset(target_val, batch_size=1024, **ns)
    data = train_dataset.prepare()
    val_data = val_dataset.prepare()

    random_latent = np.random.normal(size=(args.count, hp.values['sampling_units']))
    sample_param = decoder(random_latent).numpy()
    sample_param = (sample_param+1)/2

    for row in sample_param:
        if args.parsable:
            print(*["{0:.6f}".format(i) for i in row], sep=',')
        else:
            print(*["{0: .6f}".format(i) for i in row], sep=', ')
