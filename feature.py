import tensorflow as tf
import numpy as np


class _Feature(collections.namedtuple("_Feature", [
    'feature_name', 'input_type', 'output_type'
])):
    def __new__(cls,
                feature_name,
                input_type,
                output_type):
        return super(_Feature, cls).__new__(
            cls,
            feature_name,
            input_type,
            output_type
        )
    
    
    def transform_feature(self, read_data):
        read_data[self.feature_name] = tf.cast(read_data[self.feature_name], self.output_type)
    

    def input_feature(self, read_feature):
        read_feature[self.feature_name] = tf.FixedLenFeature([], dtype=self.input_type)
    
    
    def export_feature(self, feature_spec):
        feature_spec[self.feature_name] = tf.placeholder(self.output_type, shape=(None, 1))


uid = _Feature("uid", tf.int64, tf.int64)

feature_set = [uid]