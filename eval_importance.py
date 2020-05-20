# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.contrib.predictor import from_saved_model
from feature import feature_set, _Feature, _INDICES, _VALUES, _SHAPE
from feature import *
import argparse
import math
import numpy as np
import os, sys

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/cos_person/training_output/save_sequence0.1/1589716454", help="model path")
parser.add_argument('--left_file', type=int, default=60)
parser.add_argument('--right_file', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--batch_num", type=int, default=1000, help="how many batch you want to evaluate")
parser.add_argument("--target", type=str, default="age", help="target")


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


def main(argv):
    read_feature = {}
    for f in feature_set:
        f.input_feature(read_feature)


    def parse(record):
        read_data = tf.parse_example(serialized=record,
                                    features=read_feature)
        for f in feature_set:
            f.transform_feature(read_data)
        
        labels = {
            "age": read_data["age"] - 1,
            "gender": tf.math.equal(read_data["gender"], tf.fill(tf.shape(read_data["gender"]), tf.constant(2, tf.int64)))
        }
        read_data.pop("gender")
        read_data.pop("age")
        return read_data, labels


    args = parser.parse_args(argv[1:])

    NOT_EXIST = _Feature("NOT_EXIST")
    model_pred_fn = from_saved_model(args.model_path, 'predict')
    whole_training_list = ["/cos_person/training_data_tfrecord_v2/train_tfrecord_{}.gz".format(i) for i in range(args.left_file,1+args.right_file)]
    accuracy = {}
    all_features = [rcid, cid]
    for current_feature in [NOT_EXIST] + all_features:
        file_iterator = tf.data.Dataset.list_files(whole_training_list)
        file_iterator.shuffle(10*1024*1024)
        ds = file_iterator.interleave(lambda file : tf.data.TFRecordDataset(file, "GZIP", 1024), cycle_length=2, block_length=1)
        # ds = tf.data.TFRecordDataset(whole_training_list, "GZIP", 1024)
        ds = ds.batch(args.batch_size)
        # ds = ds.shuffle(10*1024*1024)
        ds = ds.map(parse, num_parallel_calls=10)
        ds = ds.prefetch(1)
        truth, pred, uids = np.array([]), np.array([]), np.array([])
        for feature_dict, label_tensor in ds.take(args.batch_num):
            truth = np.concatenate((truth, label_tensor[args.target].numpy()))
            uids = np.concatenate((uids, feature_dict['user_id'].numpy()))
            for f in feature_dict:
                if f == current_feature.feature_name:
                    feature_dict[f] = tf.reshape(tf.random_shuffle(feature_dict[f]), [-1,1]).numpy()
                elif f == current_feature.feature_name + _VALUES:
                    feature_dict[f] = tf.reshape(tf.random_shuffle(feature_dict[f]), [-1,]).numpy()
                elif f.endswith(_VALUES):
                    feature_dict[f] = tf.reshape(feature_dict[f], [-1,]).numpy()
                elif f.endswith(_INDICES):
                    feature_dict[f] = tf.reshape(feature_dict[f], [-1, 2]).numpy()
                elif f.endswith(_SHAPE):
                    feature_dict[f] = tf.reshape(feature_dict[f], [-1,]).numpy()
                else:
                    feature_dict[f] = tf.reshape(feature_dict[f], [-1,1]).numpy()

            res = model_pred_fn(feature_dict)['{}/class_ids'.format(args.target)].flatten()
            pred = np.concatenate((pred, res))

        matches = tf.equal(pred, tf.cast(truth, tf.int64))
        accuracy[current_feature.feature_name] = tf.reduce_mean(tf.cast(matches, tf.float32))
        print("Shuffling feature {} got accuracy {:.4f}".format(current_feature.feature_name, accuracy[current_feature.feature_name]))
    print("Result:")
    for f in accuracy:
        print("{}\t{:.4f}".format(f, accuracy[f]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)