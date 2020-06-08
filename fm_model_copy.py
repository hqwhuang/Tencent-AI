# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.feature_column import numeric_column, embedding_column, shared_embedding_columns, indicator_column, bucketized_column
from tensorflow.contrib.layers import sparse_column_with_integerized_feature, \
                                      sparse_column_with_hash_bucket
from feature import fm_set_v1, fm_set_v2, fm_set_v3
import argparse
import math
import os, sys, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--model_name', type=str, default="sequence")
parser.add_argument('--model_dir', type=str, default="/cos_person/training_output/model_fm_copy")
parser.add_argument('--save_dir', type=str, default="/cos_person/training_output/save_fm_copy")
parser.add_argument('--per_file', type=int, default=12)
parser.add_argument('--last_file', type=int, default=2) #121
parser.add_argument("--train_path", type=str, default="/cos_person/kfold/neg_tfrecord")
parser.add_argument("--debug_mode", action="store_true", default=False)


TEMP = "temp/"
CID_EMBEDDING_DIMENSION = 64
combiner_map = {
    "sequencev1": "sum",
    "sequencev2": "sum",
    "sequencev3": "sum"
}

feature_set_map = {
    "sequencev1": fm_set_v1,
    "sequencev2": fm_set_v1,
    "sequencev3": fm_set_v1
}

lr_map = {
    "sequencev1": 1,
    "sequencev2": 0.1,
    "sequencev3": 0.01
}

threshold_map = {
    "sequencev1": 20,
    "sequencev2": 15,
    "sequencev3": 10
}


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
    args = parser.parse_args(argv[1:])
    label_weights = {
        "gender": 0.5
    }
    read_feature = {}
    for f in feature_set_map[args.model_name]:
        f.input_feature(read_feature)


    def parse(record):
        read_data = tf.parse_example(serialized=record,
                                    features=read_feature)
        for f in feature_set_map[args.model_name]:
            f.transform_feature(read_data)
        
        labels = tf.math.equal(read_data["gender"], tf.fill(tf.shape(read_data["gender"]), tf.constant(2, tf.int64)))
        read_data.pop("gender")
        return read_data, labels
    

    def filter_parse(record):
        read_data = tf.parse_single_example(serialized=record,
                                            features=read_feature)
        return read_data

    
    def filter_check(read_data):
        return read_data['i'] <= threshold_map[args.model_name]


    def filter_map(read_data):
        for f in feature_set_map[args.model_name]:
            f.transform_feature(read_data)
        
        labels = tf.math.equal(read_data["gender"], tf.fill(tf.shape(read_data["gender"]), tf.constant(2, tf.int64)))
        read_data.pop("gender")
        return read_data, labels


    def get_input_fn(filenames, epoch=1, batch_size=128, compression="GZIP"):
        def input_fn():
            ds = tf.data.TFRecordDataset(filenames, compression, 10*1024*1024*1024)
            ds = ds.repeat(epoch).batch(batch_size)
            ds = ds.map(parse, num_parallel_calls=10)
            ds = ds.prefetch(buffer_size=20)
            return ds
        
        def input_fn_filter():
            ds = tf.data.TFRecordDataset(filenames, compression, 2*1024*1024*1024)
            ds = ds.map(filter_parse, num_parallel_calls=10)
            ds = ds.filter(filter_check)
            ds = ds.repeat(epoch).batch(batch_size)
            ds = ds.map(filter_map, num_parallel_calls=10)
            ds = ds.prefetch(buffer_size=5)
            return ds


        return input_fn


    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        weight_columns = {x: numeric_column(x+"_weight") for x in label_weights.keys()}
        head = tf.contrib.estimator.binary_classification_head(name="gender")
        temp = {'creative_id': features['creative_id']}
        temp['rcid'] = tf.sparse.SparseTensor(features['rcid_indices'], features['rcid_values'], features['rcid_shape'])

        with tf.variable_scope("fm"):
            cid_rcid_feature_column = params['sequence_feature_columns'].pop('rcid')
            cid_rcid_embedding = tf.feature_column.input_layer(temp, cid_rcid_feature_column)
            cid_embedding, rcid_embedding = tf.split(cid_rcid_embedding, [CID_EMBEDDING_DIMENSION, CID_EMBEDDING_DIMENSION], axis=1)

            # rcid_up, rcid_bias = tf.split(rcid_embedding, [CID_EMBEDDING_DIMENSION-1, 1], axis=1)
            # cid_up, cid_bias = tf.split(cid_embedding, [CID_EMBEDDING_DIMENSION-1, 1], axis=1)

            # rcid_cid_cross = tf.reduce_sum(tf.multiply(rcid_up, cid_up), axis=1, keepdims=True)

            # fm_logits = tf.add_n([rcid_cid_cross, rcid_bias, cid_bias])
            rcid_cid_cross = tf.reduce_sum(tf.multiply(cid_embedding, rcid_embedding), axis=1, keepdims=True)
            fm_logits = tf.add_n([rcid_cid_cross])

            tf.logging.info("fm_logits shape={}".format(fm_logits.shape))
        
        if is_training:
            tf.summary.histogram("fm_output/rcid_cid_cross", rcid_cid_cross)
            # tf.summary.histogram("fm_output/linear_bias", rcid_bias + cid_bias)
            tf.summary.histogram("sample/gender", tf.cast(labels, tf.int32))
        fm_optimizer = tf.train.AdagradOptimizer(learning_rate=lr_map[args.model_name])

        def _train_op_fn(loss):
            train_ops = []
            tf.summary.scalar('loss', loss)
            train_ops.append(fm_optimizer.minimize(loss=loss, 
                                                    global_step=tf.train.get_global_step(),
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fm")))
            return control_flow_ops.group(*train_ops)
        

        return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            logits=fm_logits,
            train_op_fn=_train_op_fn
        )

    def get_classifier():
        sequence_feature_columns = {}


        def _get_shared_sequence_column(feature1, feature2, bucket_size, dimension, combiner="sum"):
            seq_categorical_column = sparse_column_with_integerized_feature(column_name=feature1,
                                bucket_size=bucket_size)
            categorical_column = sparse_column_with_integerized_feature(column_name=feature2,
                                        bucket_size=bucket_size)
            sequence_feature_columns[feature1] = shared_embedding_columns([seq_categorical_column, categorical_column], dimension=dimension,
                                            combiner=combiner, initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(dimension)))


        _get_shared_sequence_column("rcid", "creative_id", bucket_size=5000000, dimension=CID_EMBEDDING_DIMENSION, combiner=combiner_map[args.model_name])


        classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=TEMP+args.model_name if not args.debug_mode else args.model_dir+args.model_name,
            params={
                'sequence_feature_columns': sequence_feature_columns,
            },
            config=tf.estimator.RunConfig(keep_checkpoint_max=1)
        )
        return classifier
    

    feature_spec = {}

    for f in feature_set_map[args.model_name]:
        f.export_feature(feature_spec)

    classifier = get_classifier()
    if args.debug_mode:
        # print(classifier.get_variable_names())
        print(classifier.get_variable_value('fm/input_layer/creative_id_rcid_shared_embedding/embedding_weights').mean())
        return

    whole_training_list = ["{}/train_tfrecord_{}.gz".format(args.train_path, i) for i in range(1,1+args.last_file)]
    
    training_list_list = []
    for i in range(0, len(whole_training_list), args.per_file):
        training_list_list.append(whole_training_list[i:i+args.per_file])

    for q in range(args.epoch):
        last_file = []
        for file_list_index, filenames in enumerate(training_list_list):
            filenames = last_file + filenames
            last_file = [filenames.pop()]
            tf.logging.info('Training file: {}'.format(filenames))

            classifier.train(
                input_fn=get_input_fn(
                    filenames,
                    epoch=1,
                    batch_size=args.batch_size
                ),
                hooks=[]
            )
            classifier.evaluate(
                input_fn=get_input_fn(
                    last_file,
                    epoch=1,
                    batch_size=args.batch_size
                ),
                name=args.model_name,
                steps=1000,
                hooks=[]
            )
        if len(last_file) != 0:
            classifier.train(
                input_fn=get_input_fn(
                    last_file,
                    epoch=args.epoch,
                    batch_size=args.batch_size
                ),
                hooks=[]
            )
    
    tf.logging.info("Finish train and evaluate")
    classifier.export_saved_model(args.save_dir+args.model_name, tf.estimator.export.build_raw_serving_input_receiver_fn(features=feature_spec))
    tf.logging.info("Finish export model")
    out_bytes = subprocess.check_output(['cp', '-r', TEMP+args.model_name, args.model_dir+args.model_name])
    tf.logging.info("Finish copy model_dir")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)