import sys
sys.path.append("/home/kaidong/Projects/PythonProjects/pkl-keras-train")

from keras_custom.prelude import init_module_cfg

import os

import pandas
import tensorflow
import numpy

def read_data(filename):
    from keras_custom.SegmentedTrajectories import processing_v, processing_e

    df = pandas.read_json(filename, lines=True)
    df.rename(columns={'context_v': 'v', 'env_param': 'e', 'group_ids': 'i'}, inplace=True)
    df['v'] = df['v'].apply(processing_v).apply(lambda x: x.tolist())
    df['e'] = df['e'].apply(processing_e).apply(lambda x: x.tolist())
    df['z'] = df['z'].apply(lambda x: x[0])
    df['gid'] = pandas.factorize(df['seq_name'])[0]
    return df

def prepare_reconstruct_set(dataframe):
    def processing_seq(seq):
        return seq.split('_')[0]
    dataframe = dataframe.copy()
    dataframe['s'] = dataframe['seq_name'].apply(processing_seq)
    v = tensorflow.RaggedTensor.from_value_rowids(numpy.stack(dataframe['v']), value_rowids=dataframe['gid'])
    # result will be different by frames, since we don't need
    # per frame information, we keep the first occurance
    uniq = dataframe.groupby('gid', sort=False).first()[['seq_name', 's', 'e', 'i']]
    return v, uniq

def get_model():
    from keras_custom.globals import HYPER_PARAMS
    from SegmentedAutoencoder import custom_builder, register_hyper_params
    from keras_hyper import update_hp_parameters

    register_hyper_params()
    update_hp_parameters(HYPER_PARAMS)

    model = custom_builder(HYPER_PARAMS)
    model.load_weights('result_model')
    return model

def reconstruct(model, v, q):
    from keras_custom.globals import MODULE_CFG

    y = model.predict(v)
    q['y'] = y.tolist()
    APARAM_SIZE = MODULE_CFG.get('APARAM_SIZE', 1)

    per_agent_data = []
    for index, row in q.iterrows():
        y = row['y']
        s = row['s']
        for idx, aid in enumerate(row['i']):
            per_agent_data.append({'s': s, 'aid': aid, 'p': y[idx*APARAM_SIZE:(idx+1)*APARAM_SIZE] })

    per_agent_df = pandas.DataFrame(per_agent_data)

    result_data = []
    for grp, scenario in per_agent_df.groupby('s'):
        params = scenario.groupby('aid').agg({"p":pandas.DataFrame.sample})
        result = params['p'].explode().to_numpy() * 0.5 + 0.5
        result = np.clip(result, 0, 1)
        result_data.append({'s': grp, 'p': result.tolist()})
    result_df = pandas.DataFrame(result_data)
    return result_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input json file')
    parser.add_argument('-C', '--direrctory', type=str, required=True,
                        help='directory of the model')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='output json file')
    args = parser.parse_args()

    os.chdir(args.direrctory)
    init_module_cfg()
    df = read_data(args.input)
    v, q = prepare_reconstruct_set(df)
    model = get_model()
    result = reconstruct(model, v, q)
    result.to_json(args.output, orient='records', lines=True)
