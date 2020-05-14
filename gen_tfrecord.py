# -*- coding: utf-8 -*- 
import argparse
import tensorflow as tf
import os, sys

parser = argparse.ArgumentParser()


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


def gen_tfrecord(argv):
    args = parser.parse_args(argv[1:])
    for i in range(1, 122):
        with open("/cos_person/training_data/train_serialize_{}.csv".format(i), "r") as f:
            lines = f.readlines()
            key = lines[0].strip().split(",")
            key_float = key[:12]
            key_int64 = key[12:]
            values = lines[1:]
            options = tf.python_io.TFRecordOptions(compression_type="GZIP", compression_level=9)
            writer = tf.python_io.TFRecordWriter("/cos_person/training_data_tfrecord/train_tfrecord_{}.gz".format(i), options=options)
            for line in values:
                value_float = line.strip().split(",")[:12]
                value_int64 = line.strip().split(",")[12:]
                whole_feature = {
                    feature_name: tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)])) for (feature_name, value) in list(zip(key_float, value_float))
                }
                whole_feature.update({
                    feature_name: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in value.split(":")])) for (feature_name, value) in list(zip(key_int64, value_int64))
                })
                tf_example = tf.train.Example(
                    features=tf.train.Features(feature=whole_feature)
                )
                writer.write(tf_example.SerializeToString())
            writer.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(gen_tfrecord)