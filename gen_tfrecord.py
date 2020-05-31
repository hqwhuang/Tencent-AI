# -*- coding: utf-8 -*- 
import argparse
import tensorflow as tf
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--left_file', type=int, default=1)
parser.add_argument('--right_file', type=int, default=24)
parser.add_argument('--input_path', type=str, default="/cos_person/training_data_v2/train_serialize_")
parser.add_argument('--output_path', type=str, default="/cos_person/training_data_tfrecord_v2/train_tfrecord_")


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


def run(index, args):
    with open("{}{}.csv".format(args.input_path, index), "r") as f:
        lines = f.readlines()
        key = lines[0].strip().split(",")
        # key_float = key[:12]
        key_int64 = key
        values = lines[1:]
        options = tf.python_io.TFRecordOptions(compression_type="GZIP", compression_level=9)
        writer = tf.python_io.TFRecordWriter("{}{}.gz".format(args.output_path, index), options=options)
        for line in values:
            # value_float = line.strip().split(",")[:12]
            value_int64 = line.strip().split(",")
            # whole_feature = {
            #     feature_name: tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)])) for (feature_name, value) in list(zip(key_float, value_float))
            # }
            whole_feature = {
                feature_name: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in value.split(":")] if value != "" else [])) for (feature_name, value) in list(zip(key_int64, value_int64))
            }
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=whole_feature)
            )
            writer.write(tf_example.SerializeToString())
        writer.close()


def gen_tfrecord(argv):
    args = parser.parse_args(argv[1:])
    for i in range(args.left_file, 1+args.right_file):
        run(i, args)
        


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(gen_tfrecord)