# -*- coding: utf-8 -*- 
import argparse
import tensorflow as tf
import numpy as np
from decimal import *
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
# parser.add_argument('--input_file', type=str,
#                     help='input file')
# parser.add_argument('--export_dir', type=str,
#                     help='path to export the tfrecord training data')


def gen_tfrecord(argv):
    args = parser.parse_args(argv[1:])
    age_stat = ['age_stat{}'.format(i) for i in range(1,11)] 
    gender_stat = ['gender_stat{}'.format(i) for i in range(1,3)] 
    age_ratio = ['age_ratio{}'.format(i) for i in range(1,11)] 
    gender_ratio = ['gender_ratio{}'.format(i) for i in range(1,3)] 

    features_int64 = age_stat + gender_stat
    features_float =  age_ratio + gender_ratio


    def _parse(record):
        read_features = {
            f: tf.FixedLenFeature([], dtype=tf.int64) for f in features_int64 #VarLenFeature
        }
        read_features.update({
            f: tf.FixedLenFeature([], dtype=tf.float32) for f in features_float
        })
        read_data = tf.parse_example(serialized=record,
                                    features=read_features)
        return read_data


    features = features_int64 + features_float
    result = {
        f: [] for f in features
    }
    for j in range(1,122):
        ds = tf.data.TFRecordDataset(["/cos_person/training_data_tfrecord/train_tfrecord_{}.gz".format(j)], 'GZIP', 1024*1024*1024)
        ds = ds.batch(128)
        ds = ds.map(_parse)
        
        for read_data in ds:
            try:
                for f in features:
                    for i in range(len(read_data[f].numpy())):
                        result[f].append(read_data[f].numpy()[i])

            except Exception as e:
                print(e)
                continue

    with open("/cos_person/output/bucket.txt", "w") as file:
        for f in features:
            result[f] = np.percentile(result[f], range(0, 100, 5))
            result[f] = sorted(list(set(result[f])))
            file.write("{}: {}\n".format(f, [float(Decimal("%.4f" % x)) for x in result[f]]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(gen_tfrecord)