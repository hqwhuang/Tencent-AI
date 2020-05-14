# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.contrib.predictor import from_saved_model
from feature import feature_set
import argparse
import math
import numpy as np

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/cos_person/training_output/save_baseline/1589262665", help="model path")
parser.add_argument('--last_file', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)


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

    model_pred_fn = from_saved_model(args.model_path, 'predict')
    whole_training_list = ["/cos_person/test_data_tfrecord/test_tfrecord_{}.gz".format(i) for i in range(1,1+args.last_file)]
    ds = tf.data.TFRecordDataset(whole_training_list, "GZIP", 1024)
    ds = ds.batch(args.batch_size)
    ds = ds.map(parse, num_parallel_calls=10)
    ds = ds.prefetch(1)
    pred, uids = np.array([]), np.array([])
    for feature_dict, label_tensor in ds:
        uids = np.concatenate((uids, feature_dict['user_id'].numpy()))
        preds_map = model_pred_fn(feature_dict)
        print(preds_map.keys())
        res = preds_map['{}/logistic'.format(args.target)].flatten()
        pred = np.concatenate((pred, res))
    with open("/cos_person/output/result.txt", "r") as f:
        f.write("{}".format(pred))
        value = tf.argmax(pred,1) if args.target == "age" else tf.cast(pred > 0.5, tf.int64)+1
        f.write("\n{}".format(value))
        f.write("\n{}".format(uids))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)