# -*- coding: utf-8 -*- 
import argparse
import tensorflow as tf
import os, sys
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--context', type=str, default="test")#train

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


def run(args):
    with open("/cos_person/second_phase/{}_serialize.csv".format(args.context), "r") as f:
        lines = f.readlines()
        key = lines[0].strip().split(",")
        key_int64 = key
        values = lines[1:]
        options = tf.python_io.TFRecordOptions(compression_type="GZIP", compression_level=9)
        writer = tf.python_io.TFRecordWriter("/cos_person/second_phase_tfrecord/{}_tfrecord.gz".format(args.context), options=options)
        for line in values:
            value_int64 = line.strip().split(",")
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
    run(args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(gen_tfrecord)