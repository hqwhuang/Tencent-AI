# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.contrib.predictor import from_saved_model
from feature import feature_set_v3, _INDICES, _VALUES, _SHAPE
import argparse
import math
import numpy as np
import os, sys
import threading

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/cos_person/training_output/save_sequencev3/1590897715", help="model path")
parser.add_argument('--left_file', type=int, default=1)
parser.add_argument('--right_file', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=128)


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
    read_feature = {}
    for f in feature_set_v3:
        f.input_feature(read_feature)


    def parse(record):
        read_data = tf.parse_example(serialized=record,
                                    features=read_feature)
        for f in feature_set_v3:
            f.transform_feature(read_data)
        
        labels = {
            "age": read_data["age"] - 1,
            "gender": tf.math.equal(read_data["gender"], tf.fill(tf.shape(read_data["gender"]), tf.constant(2, tf.int64)))
        }
        read_data.pop("gender")
        read_data.pop("age")
        return read_data, labels


    model_pred_fn = from_saved_model(args.model_path, 'predict')
    whole_training_list = ["/cos_person/testing_data_tfrecord_v2/test_tfrecord_{}.gz".format(i) for i in range(index,1+index)]
    # whole_training_list = ["/cos_person/training_data_tfrecord/train_tfrecord_{}.gz".format(i) for i in range(1,1+args.last_file)]
    ds = tf.data.TFRecordDataset(whole_training_list, "GZIP", 1024)
    ds = ds.batch(args.batch_size)
    ds = ds.map(parse, num_parallel_calls=10)
    ds = ds.prefetch(1)
    age_pred, gender_pred, uids, cids = np.array([]), np.array([]), np.array([]), np.array([])
    for feature_dict, label_tensor in ds:
        uids = np.concatenate((uids, feature_dict['user_id'].numpy()))
        cids = np.concatenate((cids, feature_dict['creative_id'].numpy()))
        for k in feature_dict:
            if k == 'user_id':
                feature_dict[k] = tf.reshape(tf.cast(tf.fill(tf.shape(feature_dict[k]), 0), tf.int64), [-1,1]).numpy()
            elif k.endswith(_VALUES):
                feature_dict[k] = tf.reshape(feature_dict[k], [-1,]).numpy()
            elif k.endswith(_INDICES):
                feature_dict[k] = tf.reshape(feature_dict[k], [-1, 2]).numpy()
            elif k.endswith(_SHAPE):
                feature_dict[k] = tf.reshape(feature_dict[k], [-1,]).numpy()
            else:
                feature_dict[k] = tf.reshape(feature_dict[k], [-1,1]).numpy()
        preds_map = model_pred_fn(feature_dict)
        age_res = preds_map['age/class_ids'].flatten()
        gender_res = preds_map['gender/class_ids'].flatten()
        age_pred = np.concatenate((age_pred, age_res))
        gender_pred = np.concatenate((gender_pred, gender_res))
    with open("/cos_person/output/predict/result_{}.txt".format(index), "w") as f:
        f.write("user_id,predicted_age,predicted_gender,cid\n")
        result = list(zip(uids, age_pred+1, gender_pred+1, cids))
        for uid, age, gender, cid in result:
            f.write("{},{},{},{}\n".format(int(uid),int(age),int(gender), int(cid)))


def main(argv):
    args = parser.parse_args(argv[1:])
    for i in range(args.left_file, 1+args.right_file):
        # x = threading.Thread(target=run, args=(i, args))
        # x.start()
        run(i, args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)