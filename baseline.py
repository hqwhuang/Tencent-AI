# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.feature_column import numeric_column, embedding_column, shared_embedding_columns, indicator_column, bucketized_column
from tensorflow.contrib.layers import sparse_column_with_integerized_feature, \
                                      sparse_column_with_hash_bucket
from feature import feature_set
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--per_file', type=int, default=12)


label_weights = {}
read_feature = {}
for f in feature_set:
    f.input_feature(read_feature)


def parse(record, label_weights):#TODO: finish
    read_data = tf.parse_example(serialized=record,
                                 features=read_feature)
    for f in feature_set:
        f.transform_feature(read_data)
    
    labels = {}
    return read_data, labels


def get_input_fn(filenames, epoch=1, batch_size=128, compression="GZIP"):
    def input_fn():
        ds = tf.data.TFRecordDataset(filenames, compression, 10*1024*1024*1024)
        ds = ds.repeat(epoch).batch(batch_size)
        ds = ds.map(parse, num_parallel_calls=10)
        ds = ds.prefetch(buffer_size=20)
    return input_fn


def model_fn(features, labels, mode, params):
    weight_columns = {x: numeric_column(x+"_weight") for x in label_weights.keys()}
    labels = list(label_weights.keys())
    labels.sort()
    head = tf.contrib.estimator.multi_head([
        tf.estimator.MultiClassHead(n_classes=10, name="age"),
        tf.estimator.BinaryClassHead(name="gender")
    ])
    with tf.variable_scope("fm"):
        pass

    with tf.variable_scope("deep"):
        pass


def main(argv):
    args = parser.parse_args(argv[1:])

    def get_classifier():
        feature_columns = {}


        def _get_column(feature, bucket_size=10, dimension=args.embedding_dimension, dtype=tf.int32):
            pass

        classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=args.model_dir,
            params={
                'feature_columns': feature_columns
            }
        )
        return classifier
    

    feature_spec = {}

    for f in feature_set:
        f.export_feature(feature_spec)

    classifier = get_classifier()

    whole_training_list = [i for i in range(1,122)]
    training_list_list = []
    for i in range(0, len(whole_training_list), args.per_file):
        training_list_list.append(whole_training_list[i:i+args.per_file])

    last_file = []
    for file_list_index, filenames in enumerate(training_list_list):
        filenames = last_file + filenames
        last_file = [filenames.pop()]
        tf.logging.info('Training file: {}'.format(filenames))

        classifier.train(
            input_fn=get_input_fn(
                filenames,
                epoch=args.epoch,
                batch_size=128
            ),
            hooks=[]
        )

        classifier.evaluate(
            input_fn=get_input_fn(
                last_file,
                epoch=1,
                batch_size=128
            ),
            name="baseline",
            steps=100
        )
    tf.logging.info("Finish train and evaluate")
    classifier.export_saved_model(args.save_dir, tf.estimator.export.build_raw_serving_input_receiver_fn(features=feature_spec))
    tf.logging.info("Finish export model")



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)