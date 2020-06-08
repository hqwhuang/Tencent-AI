# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.feature_column import numeric_column, embedding_column, shared_embedding_columns, indicator_column, bucketized_column
from tensorflow.contrib.layers import sparse_column_with_integerized_feature, \
                                      sparse_column_with_hash_bucket
from feature import feature_set_tot
import argparse
import math
import os, sys, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--model_name', type=str, default="sequence")
parser.add_argument('--model_dir', type=str, default="/cos_person/training_output/model_pretrain_")
parser.add_argument('--save_dir', type=str, default="/cos_person/training_output/save_pretrain_")
parser.add_argument('--per_file', type=int, default=12)
parser.add_argument('--last_file', type=int, default=2) #121
parser.add_argument("--hidden_layers", type=str, default="64,32")
parser.add_argument("--train_path", type=str, default="/cos_person/training_data_tfrecord_v2")
parser.add_argument("--test_path", type=str, default="/cos_person/training_data_tfrecord_v2")
parser.add_argument('--test_last_file', type=int, default=2) #121
parser.add_argument("--debug_mode", action="store_true", default=False)
parser.add_argument("--evaluate_on", action="store_true", default=False)


TEMP = "temp/"
CID_EMBEDDING_DIMENSION = 64
combiner_map = {
    "sequencev1": "sum",
    "sequencev2": "sum",
    "sequencev3": "sum",
}

feature_set_map = {
    "sequencev1": feature_set_tot,
    "sequencev2": feature_set_tot,
    "sequencev3": feature_set_tot
}

checkpoint_map = {
    "sequencev1": "/cos_person/training_output/model_fm_sequencev1",
    "sequencev2": "/cos_person/training_output/model_fm_sequencev1",
    "sequencev3": "/cos_person/training_output/model_fm_sequencev1",
    # "sequencev3": "/cos_person/training_output/model_fm_copysequencev3/"
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
        "age": 0.5,
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
        
        labels = {
            "age": read_data["age"] - 1,
            "gender": tf.math.equal(read_data["gender"], tf.fill(tf.shape(read_data["gender"]), tf.constant(2, tf.int64)))
        }
        read_data["age_weight"] = tf.fill(tf.shape(read_data["age"]), 1.0)
        read_data["gender_weight"] = tf.fill(tf.shape(read_data["gender"]), 1.0)
        read_data.pop("gender")
        read_data.pop("age")
        return read_data, labels


    def get_input_fn(filenames, epoch=1, batch_size=128, compression="GZIP"):
        def input_fn():
            ds = tf.data.TFRecordDataset(filenames, compression, 10*1024*1024*1024)
            ds = ds.repeat(epoch).batch(batch_size)
            ds = ds.map(parse, num_parallel_calls=10)
            ds = ds.prefetch(buffer_size=20)
            return ds
        return input_fn


    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        weight_columns = {x: numeric_column(x+"_weight") for x in label_weights.keys()}
        head = tf.contrib.estimator.multi_head([
            tf.contrib.estimator.multi_class_head(n_classes=10, name="age"),
            tf.contrib.estimator.binary_classification_head(name="gender")
        ])
        temp = {'creative_id': features['creative_id']}
        temp['rcid'] = tf.sparse.SparseTensor(features['rcid_indices'], features['rcid_values'], features['rcid_shape'])

        with tf.variable_scope("deep"):
            cid_rcid_feature_column = params['sequence_feature_columns'].pop('rcid')
            cid_rcid_embedding = tf.feature_column.input_layer(temp, cid_rcid_feature_column)
            feature_columns = list(params['feature_columns'].keys())
            feature_columns.sort()
            feature_weight = []
            for feature in feature_columns:
                feature_weight.append(tf.feature_column.input_layer(features, params['feature_columns'][feature]))
            net = tf.concat(feature_weight + [cid_rcid_embedding], axis=1, name="deep_input_layer")
            tf.logging.info("deep input shape={}".format(net.shape))
            tf.summary.histogram('input', net)
            cnt = 0
            layer_name = "dense"
            for units in params['hidden_layers']:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
                with tf.variable_scope(layer_name, reuse=True):
                    weights = tf.get_variable("kernel")
                    bias = tf.get_variable("bias")
                tf.summary.histogram('{}_kernel'.format(units), weights)
                tf.summary.histogram('{}_bias'.format(units), bias)
                tf.summary.histogram('{}'.format(units), net)
                cnt += 1
                layer_name= "dense_{}".format(cnt)
            
            dense_logits = tf.layers.dense(net, units=head.logits_dimension, activation=None, name="dense_logit")
            global_bias = tf.get_variable(name="global_bias", shape=[head.logits_dimension], initializer=tf.constant_initializer(0.0))

            logits = tf.nn.bias_add(dense_logits, global_bias)
        
        if is_training:
            tf.summary.histogram("sample/gender", tf.cast(labels["gender"], tf.int32))
            tf.summary.histogram("sample/age", labels["age"])
        deep_optimizer = tf.train.AdagradOptimizer(learning_rate=0.005)

        def _train_op_fn(loss):
            train_ops = []
            l2_loss = tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', loss)
            train_ops.append(deep_optimizer.minimize(loss=loss + l2_loss, 
                                                    global_step=tf.train.get_global_step(),
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="deep")))
            return control_flow_ops.group(*train_ops)
        

        return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            logits=logits,
            train_op_fn=_train_op_fn
        )

    def get_classifier():
        feature_columns = {}
        sequence_feature_columns = {}

        def _get_embedding_column(feature, dimension, bucket_size=10, dtype=tf.int64):
            categorical_column = sparse_column_with_integerized_feature(column_name=feature,
                                    bucket_size=bucket_size, dtype=dtype)
            
            feature_columns[feature] = embedding_column(categorical_column, dimension=dimension, initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(dimension)))


        def _get_stat_column(feature, boundaries):
            feature_columns[feature] = indicator_column(bucketized_column(numeric_column(feature), boundaries))
        
        
        def _get_ratio_column(feature):
            categorical_column = sparse_column_with_integerized_feature(column_name=feature,
                                    bucket_size=101, dtype=tf.int64)
            feature_columns[feature] = indicator_column(categorical_column)


        def _get_shared_sequence_column(feature1, feature2, bucket_size, dimension, combiner="sum"):
            seq_categorical_column = sparse_column_with_integerized_feature(column_name=feature1,
                                bucket_size=bucket_size)
            categorical_column = sparse_column_with_integerized_feature(column_name=feature2,
                                        bucket_size=bucket_size)
            sequence_feature_columns[feature1] = shared_embedding_columns([seq_categorical_column, categorical_column], dimension=dimension,
                                            combiner=combiner,
                                            ckpt_to_load_from=checkpoint_map[args.model_name], tensor_name_in_ckpt="fm/input_layer/creative_id_rcid_shared_embedding/embedding_weights", trainable=False)


        _get_shared_sequence_column("rcid", "creative_id", bucket_size=5000000, dimension=CID_EMBEDDING_DIMENSION, combiner=combiner_map[args.model_name])
        # _get_embedding_column("user_id", 8, bucket_size=1000000)
        _get_embedding_column("time", 3, bucket_size=7)
        # _get_embedding_column("creative_id", 8, bucket_size=5000000)
        _get_embedding_column("ad_id", 8, bucket_size=4000000)
        _get_embedding_column("product_id", 8, bucket_size=100000)
        _get_embedding_column("product_category", 3, bucket_size=100)
        _get_embedding_column("advertiser_id", 8, bucket_size=100000)
        _get_embedding_column("industry", 3, bucket_size=5000)

        _get_stat_column("pv", [1.0, 3.0, 5.0, 8.0, 14.0, 22.0, 36.0, 57.0, 87.0, 138.0, 213.0, 360.0, 602.0, 1039.0, 1715.0, 3271.0, 7049.0, 15514.0, 36549.0])
        _get_ratio_column("gender_ratio1")
        _get_ratio_column("gender_ratio2")
        _get_ratio_column("age_ratio1")
        _get_ratio_column("age_ratio2")
        _get_ratio_column("age_ratio3")
        _get_ratio_column("age_ratio4")
        _get_ratio_column("age_ratio5")
        _get_ratio_column("age_ratio6")
        _get_ratio_column("age_ratio7")
        _get_ratio_column("age_ratio8")
        _get_ratio_column("age_ratio9")
        _get_ratio_column("age_ratio10")

        classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=TEMP+args.model_name if not args.debug_mode else args.model_dir+args.model_name,
            params={
                'feature_columns': feature_columns,
                'sequence_feature_columns': sequence_feature_columns,
                'hidden_layers': [int(x.strip()) for x in args.hidden_layers.strip().split(",")]
            },
            config=tf.estimator.RunConfig(keep_checkpoint_max=2)
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
    test_list = ["{}/test_tfrecord_{}.gz".format(args.test_path, i) for i in range(1,1+args.test_last_file)]
    
    training_list_list = []
    for i in range(0, len(whole_training_list), args.per_file):
        training_list_list.append(whole_training_list[i:i+args.per_file])

    for q in range(args.epoch):
        for file_list_index, filenames in enumerate(training_list_list):
            filenames = filenames
            tf.logging.info('Training file: {}'.format(filenames))

            classifier.train(
                input_fn=get_input_fn(
                    filenames,
                    epoch=1,
                    batch_size=128
                ),
                hooks=[]
            )
            if args.evaluate_on:
                classifier.evaluate(
                    input_fn=get_input_fn(
                        test_list[-1],
                        epoch=1,
                        batch_size=128
                    ),
                    name=args.model_name,
                    steps=1000,
                    hooks=[]
                )
        if args.evaluate_on:
            classifier.evaluate(
                input_fn=get_input_fn(
                    test_list,
                    epoch=1,
                    batch_size=128
                ),
                name=args.model_name,
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