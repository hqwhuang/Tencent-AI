# -*- coding: utf-8 -*- 
import argparse
import tensorflow as tf
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--input_index', type=int,
                    help='input file')
# parser.add_argument('--export_dir', type=str,
#                     help='path to export the tfrecord training data')


def test_tfrecord(argv):


    def _parse(record):
        read_features = {
            f: tf.FixedLenFeature([], dtype=tf.int64) for f in ['user_id', 'pv', 'gender_stat1', 'creative_id', 'ad_id'] #VarLenFeature
        }
        read_features.update({
            f: tf.FixedLenFeature([], dtype=tf.float32) for f in ['age_ratio10', 'gender_ratio2']
        })
        read_data = tf.parse_example(serialized=record,
                                    features=read_features)
        return read_data
    

    args = parser.parse_args(argv[1:])
    ds = tf.data.TFRecordDataset(["/cos_person/training_data_tfrecord/train_tfrecord_{}.gz".format(args.input_index)], 'GZIP', 100*1024*1024)
    ds = ds.batch(2)
    ds = ds.map(_parse)
    cnt = 0
    for read_data in ds.take(1):
        print(read_data)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(test_tfrecord)