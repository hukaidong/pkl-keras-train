import pandas
import numpy
import tensorflow as tf
import os

from keras_custom.globals import MODULE_CFG

def processing_v(v):
    context_np = numpy.asarray(v, dtype=numpy.float32)
    context_np = context_np.reshape((MODULE_CFG['nframe'], -1, MODULE_CFG['nhead']))
    pad_width = ((0, 0), (0, MODULE_CFG['nagent'] - context_np.shape[1]), (0, 0))
    return numpy.pad(context_np, pad_width).flatten()


def processing_e(e):
    return numpy.asarray(e)


def prepare_slice(dataframe):
    v = tf.RaggedTensor.from_value_rowids(numpy.stack(dataframe['v']), value_rowids=dataframe['gid'])
    e = dataframe.groupby('gid').agg({'e': 'first'})
    # Pandas surrounds the values with an extra array object, so we need to remove it.
    e = numpy.stack([x[0] for x in e.values])
    return v, e


class SegmentedTrajectories:
    def __init__(self, json_path):
        assert os.path.exists(json_path)
        df = pandas.read_json(json_path, lines=True)
        df.rename(columns={'context_v': 'v', 'env_param': 'e'}, inplace=True)
        df['v'] = df['v'].apply(processing_v)
        df['e'] = df['e'].apply(processing_e)
        df['gid'] = pandas.factorize(df['seq_name'])[0]
        self.fpath = json_path
        self.dataframe = df
        self.dataset = tf.data.Dataset.from_tensor_slices(prepare_slice(df))

    def prepare(self, shuffle=False, batch_size=256):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = self.dataset
        ds = ds.repeat(10)

        if shuffle:
            ds = ds.shuffle(5000)

        ds = ds.batch(batch_size)

        return ds.prefetch(buffer_size=AUTOTUNE)

    def __str__(self):
        return f"SegmentedTrajectories( {self.fpath} )"
