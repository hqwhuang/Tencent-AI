# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.feature_column import numeric_column, embedding_column, shared_embedding_columns, indicator_column, bucketized_column
from tensorflow.contrib.layers import sparse_column_with_integerized_feature, \
                                      sparse_column_with_hash_bucket
from feature import feature_set
import argparse
import math
import os, sys, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=24)
parser.add_argument('--model_name', type=str, default="sequence")
parser.add_argument('--model_dir', type=str, default="/cos_person/training_output/model_")
parser.add_argument('--save_dir', type=str, default="/cos_person/training_output/save_")
parser.add_argument('--per_file', type=int, default=48)
parser.add_argument('--last_file', type=int, default=2) #121
parser.add_argument("--hidden_layers", type=str, default="128,64")
parser.add_argument("--lr", type=float, default=0.1)

TEMP = "temp/"
CID_EMBEDDING_DIMENSION = 128


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

        with tf.variable_scope("fm"):
            cid_rcid_feature_column = params['sequence_feature_columns'].pop('rcid')
            cid_rcid_embedding = tf.feature_column.input_layer(temp, cid_rcid_feature_column)
            cid_embedding, rcid_embedding = tf.split(cid_rcid_embedding, [CID_EMBEDDING_DIMENSION, CID_EMBEDDING_DIMENSION], axis=1)

            rcid_up, rcid_bias = tf.split(rcid_embedding, [CID_EMBEDDING_DIMENSION-1, 1], axis=1)
            cid_up, cid_bias = tf.split(cid_embedding, [CID_EMBEDDING_DIMENSION-1, 1], axis=1)

            rcid_cid_cross = tf.reduce_sum(tf.multiply(rcid_up, cid_up), axis=1, keepdims=True)

            fm_output = tf.add_n([rcid_cid_cross, rcid_bias, cid_bias])
            tf.logging.info("fm_output shape={}".format(fm_output.shape))


        with tf.variable_scope("deep"):
            feature_columns = list(params['feature_columns'].keys())
            feature_columns.sort()
            feature_weight = []
            for feature in feature_columns:
                feature_weight.append(tf.feature_column.input_layer(features, params['feature_columns'][feature]))
            net = tf.concat(feature_weight + [rcid_up, cid_up], axis=1, name="deep_input_layer")
            tf.logging.info("deep input shape={}".format(net.shape))
            tf.summary.histogram('input', net)
            cnt = 0
            layer_name = "dense"
            for units in params['hidden_layers']:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
                with tf.variable_scope(layer_name, reuse=True):
                    weights = tf.get_variable("kernel")
                    bias = tf.get_variable("bias")
                tf.summary.histogram('{}_kernel'.format(units), weights)
                tf.summary.histogram('{}_bias'.format(units), bias)
                tf.summary.histogram('{}'.format(units), net)
                cnt += 1
                layer_name= "dense_{}".format(cnt)
            
            fm_logits = tf.layers.dense(fm_output, units=head.logits_dimension, activation=None, name="fm_logits")
            dense_logits = tf.layers.dense(net, units=head.logits_dimension, activation=None, name="dense_logit")
            global_bias = tf.get_variable(name="global_bias", shape=[head.logits_dimension], initializer=tf.constant_initializer(0.0))

            logits = fm_logits + dense_logits + global_bias
        
        if is_training:
            tf.summary.histogram("fm_output/rcid_cid_cross", rcid_cid_cross)
            tf.summary.histogram("fm_output/linear_bias", rcid_bias + cid_bias)
            tf.summary.histogram("sample/gender", tf.cast(labels["gender"], tf.int32))
            tf.summary.histogram("sample/age", labels["age"])
        deep_optimizer = tf.train.AdagradOptimizer(learning_rate=0.005)
        fm_optimizer = tf.train.FtrlOptimizer(learning_rate=args.lr)

        def _train_op_fn(loss):
            train_ops = []
            tf.summary.scalar('loss', loss)
            train_ops.append(deep_optimizer.minimize(loss=loss, 
                                                    global_step=tf.train.get_global_step(),
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="deep")))
            train_ops.append(fm_optimizer.minimize(loss=loss, 
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fm")))
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
        

        def _get_shared_sequence_column(feature1, feature2, bucket_size, dimension, combiner="sum"):
            seq_categorical_column = sparse_column_with_integerized_feature(column_name=feature1,
                                bucket_size=bucket_size)
            categorical_column = sparse_column_with_integerized_feature(column_name=feature2,
                                        bucket_size=bucket_size)
            sequence_feature_columns[feature1] = shared_embedding_columns([seq_categorical_column, categorical_column], dimension=dimension,
                                            combiner=combiner, initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(dimension)))


        _get_shared_sequence_column("rcid", "creative_id", bucket_size=5000000, dimension=CID_EMBEDDING_DIMENSION, combiner="sum" if args.model_name != "sequencev2" else "sqrtn")
        # _get_embedding_column("user_id", 8, bucket_size=1000000)
        _get_embedding_column("time", 3, bucket_size=7)
        # _get_embedding_column("creative_id", 8, bucket_size=5000000)
        _get_embedding_column("ad_id", 8, bucket_size=4000000)
        _get_embedding_column("product_id", 8, bucket_size=100000)
        _get_embedding_column("product_category", 3, bucket_size=100)
        _get_embedding_column("advertiser_id", 8, bucket_size=100000)
        _get_embedding_column("industry", 3, bucket_size=5000)
        _get_stat_column("age_stat1", [0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 17.0, 34.0, 63.0, 124.0, 236.0, 581.0, 1272.0])
        _get_stat_column("age_stat2", [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 11.0, 19.0, 30.0, 53.0, 93.0, 165.0, 286.0, 539.0, 1117.0, 2258.0, 6182.0])
        _get_stat_column("age_stat3", [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 12.0, 18.0, 29.0, 46.0, 76.0, 134.0, 225.0, 398.0, 664.0, 1432.0, 3414.0, 7531.0])
        _get_stat_column("age_stat4", [0.0, 1.0, 2.0, 3.0, 5.0, 9.0, 14.0, 22.0, 34.0, 56.0, 102.0, 170.0, 279.0, 494.0, 1020.0, 2494.0, 5384.0])
        _get_stat_column("age_stat5", [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 18.0, 29.0, 49.0, 82.0, 139.0, 235.0, 450.0, 962.0, 2522.0, 4174.0])
        _get_stat_column("age_stat6", [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 33.0, 57.0, 101.0, 165.0, 320.0, 567.0, 1720.0, 3588.0])
        _get_stat_column("age_stat7", [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 12.0, 20.0, 35.0, 59.0, 101.0, 175.0, 406.0, 1101.0, 2811.0])
        _get_stat_column("age_stat8", [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 14.0, 24.0, 43.0, 81.0, 192.0, 549.0, 1534.0])
        _get_stat_column("age_stat9", [0.0, 1.0, 2.0, 4.0, 6.0, 13.0, 24.0, 44.0, 103.0, 328.0, 1056.0])
        _get_stat_column("age_stat10", [0.0, 1.0, 2.0, 3.0, 6.0, 12.0, 28.0, 55.0, 173.0, 703.0])
        _get_stat_column("gender_stat1", [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 33.0, 52.0, 84.0, 134.0, 220.0, 370.0, 607.0, 1062.0, 1981.0, 4064.0, 9633.0, 25791.0])
        _get_stat_column("gender_stat2", [0.0, 1.0, 2.0, 3.0, 6.0, 9.0, 15.0, 26.0, 45.0, 74.0, 129.0, 240.0, 441.0, 757.0, 1563.0, 2899.0, 9077.0])
        _get_stat_column("pv", [1.0, 3.0, 5.0, 8.0, 14.0, 22.0, 36.0, 57.0, 87.0, 138.0, 213.0, 360.0, 602.0, 1039.0, 1715.0, 3271.0, 7049.0, 15514.0, 36549.0])
        _get_stat_column("gender_ratio1", [0.0, 0.0293, 0.2353, 0.4052, 0.5, 0.5595, 0.6016, 0.6323, 0.6665, 0.687, 0.7095, 0.75, 0.7931, 0.8414, 0.9, 0.9428, 0.9771, 0.9947, 1.0])
        _get_stat_column("gender_ratio2", [0.0, 0.0053, 0.0229, 0.0572, 0.1, 0.1586, 0.2069, 0.25, 0.2905, 0.313, 0.3335, 0.3677, 0.3984, 0.4405, 0.5, 0.5948, 0.7647, 0.9707])
        _get_stat_column("age_ratio1", [0.0, 0.0048, 0.012, 0.0169, 0.0211, 0.0254, 0.0305, 0.0357, 0.0411, 0.0472, 0.0556, 0.0646, 0.0873, 0.1477])
        _get_stat_column("age_ratio2", [0.0, 0.0376, 0.069, 0.093, 0.1071, 0.1198, 0.1324, 0.1415, 0.1536, 0.1691, 0.187, 0.2064, 0.2308, 0.25, 0.285, 0.3333, 0.4286])
        _get_stat_column("age_ratio3", [0.0, 0.0828, 0.125, 0.1538, 0.1687, 0.1808, 0.1907, 0.2006, 0.2103, 0.2222, 0.2344, 0.25, 0.2632, 0.2812, 0.3038, 0.3333, 0.375, 0.4615])
        _get_stat_column("age_ratio4", [0.0, 0.0417, 0.0837, 0.1098, 0.1254, 0.1399, 0.1465, 0.1552, 0.1612, 0.166, 0.1729, 0.1811, 0.1905, 0.2018, 0.2195, 0.2391, 0.2716, 0.3333])
        _get_stat_column("age_ratio5", [0.0, 0.0588, 0.0769, 0.0962, 0.1111, 0.1234, 0.13, 0.1391, 0.1455, 0.1526, 0.158, 0.1659, 0.1751, 0.1892, 0.2081, 0.25, 0.3333])
        _get_stat_column("age_ratio6", [0.0, 0.0097, 0.032, 0.0586, 0.0722, 0.0835, 0.0952, 0.1031, 0.1127, 0.1224, 0.1286, 0.1353, 0.1429, 0.1543, 0.1722, 0.2016, 0.2812])
        _get_stat_column("age_ratio7", [0.0, 0.0045, 0.0173, 0.0313, 0.0406, 0.0526, 0.0608, 0.0696, 0.0769, 0.084, 0.0927, 0.0994, 0.1105, 0.124, 0.1471, 0.2])
        _get_stat_column("age_ratio8", [0.0, 0.0017, 0.0067, 0.0123, 0.0179, 0.0247, 0.0319, 0.0372, 0.0435, 0.0477, 0.0533, 0.0625, 0.0789, 0.1094])
        _get_stat_column("age_ratio9", [0.0, 0.0014, 0.0043, 0.0075, 0.0134, 0.0187, 0.0239, 0.03, 0.0331, 0.0403, 0.0519, 0.0809])
        _get_stat_column("age_ratio10", [0.0, 0.0009, 0.0035, 0.0066, 0.0107, 0.0153, 0.0184, 0.0246, 0.0355, 0.0556])

        classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=TEMP+args.model_name,
            params={
                'feature_columns': feature_columns,
                'sequence_feature_columns': sequence_feature_columns,
                'hidden_layers': [int(x.strip()) for x in args.hidden_layers.strip().split(",")]
            },
            config=tf.estimator.RunConfig(keep_checkpoint_max=2)
        )
        return classifier
    

    feature_spec = {}

    for f in feature_set:
        f.export_feature(feature_spec)

    classifier = get_classifier()

    whole_training_list = ["/cos_person/training_data_tfrecord_v2/train_tfrecord_{}.gz".format(i) for i in range(1,1+args.last_file)]
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
            name=args.model_name,
            steps=100,
            hooks=[]
        )
    if len(last_file) != 0:
        classifier.train(
            input_fn=get_input_fn(
                last_file,
                epoch=args.epoch,
                batch_size=128
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