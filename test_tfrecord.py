# -*- coding: utf-8 -*- 
import argparse
import tensorflow as tf
from feature import feature_set
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--input_index', type=int,
                    help='input file')
# parser.add_argument('--export_dir', type=str,
#                     help='path to export the tfrecord training data')


def test_tfrecord(argv):


    read_feature = {}
    for f in feature_set:
        f.input_feature(read_feature)


    def _parse(record):
        read_data = tf.parse_example(serialized=record,
                                    features=read_feature)
        for f in feature_set:
            f.transform_feature(read_data)
        
        labels = {
            "age": read_data["age"] - 1,
            "gender": tf.math.equal(read_data["gender"], tf.fill(tf.shape(read_data["gender"]), tf.constant(2, tf.int64)))
        }
        read_data["age_weight"] = tf.fill(tf.shape(read_data["age"]), 1.0)
        read_data["gender_weight"] = tf.fill(tf.shape(read_data["gender"]), 1.0)
        read_data.pop("gender")
        read_data.pop("age")

        return read_data, labels
    

    args = parser.parse_args(argv[1:])
    ds = tf.data.TFRecordDataset(["/cos_person/training_data_tfrecord/train_tfrecord_{}.gz".format(args.input_index)], 'GZIP', 100*1024*1024)
    ds = ds.batch(2)
    ds = ds.map(_parse)
    cnt = 0
    for read_data, labels in ds.take(1):
        print(read_data)
        print(labels)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(test_tfrecord)