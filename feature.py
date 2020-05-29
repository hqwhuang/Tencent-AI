import tensorflow as tf
import numpy as np
import collections


_INDICES = '_indices'
_VALUES = '_values'
_SHAPE = '_shape'

class _Feature(collections.namedtuple("_Feature", [
    'feature_name', 'sequence_len', 'input_type', 'output_type'
])):
    def __new__(cls,
                feature_name,
                sequence_len=60,
                input_type=tf.int64,
                output_type=tf.int64):
        return super(_Feature, cls).__new__(
            cls,
            feature_name,
            sequence_len,
            input_type,
            output_type
        )
    
    
    def transform_feature(self, read_data):
        read_data[self.feature_name] = tf.cast(read_data[self.feature_name], self.output_type)
    

    def input_feature(self, read_feature):
        read_feature[self.feature_name] = tf.FixedLenFeature([], dtype=self.input_type)
    
    
    def export_feature(self, feature_spec):
        feature_spec[self.feature_name] = tf.placeholder(self.output_type, shape=(None, 1))


class _Label(_Feature):

    def export_feature(self, feature_spec):
        pass


class _SequenceFeature(_Feature):
    
    def transform_feature(self, read_data):
        dense_tensor = tf.sparse_to_dense(read_data[self.feature_name].indices, 
                                          tf.shape(read_data[self.feature_name], out_type=tf.int64), 
                                          read_data[self.feature_name].values,
                                          default_value=0)
        slice_tensor = dense_tensor[:, 0:self.sequence_len]
        read_data[self.feature_name+_INDICES] = tf.where(tf.not_equal(slice_tensor, 0))
        read_data[self.feature_name+_VALUES] = tf.gather_nd(slice_tensor, read_data[self.feature_name+_INDICES])
        read_data[self.feature_name+_SHAPE] = tf.shape(slice_tensor, out_type=tf.int64)
        read_data.pop(self.feature_name)


    def export_feature(self, feature_spec):
        feature_spec[self.feature_name+_INDICES] = tf.placeholder(self.output_type, shape=(None, 2)) 
        feature_spec[self.feature_name+_VALUES] = tf.placeholder(self.output_type, shape=(None, )) 
        feature_spec[self.feature_name+_SHAPE] = tf.placeholder(self.output_type, shape=(2, ))
    

    def input_feature(self, read_feature):
        read_feature[self.feature_name] = tf.VarLenFeature(dtype=self.input_type)


age = _Label("age")
gender = _Label("gender")
rcid = _SequenceFeature("rcid")
rcid_1 = _SequenceFeature("rcid", sequence_len=1)
rcid_5 = _SequenceFeature("rcid", sequence_len=5)
rcid_10 = _SequenceFeature("rcid", sequence_len=2)
rcid_second_phase = _SequenceFeature("creative_ids", sequence_len=100000)
uid = _Feature("user_id")
time = _Feature("time")
cid = _Feature("creative_id")
aid = _Feature("ad_id")
product_id = _Feature("product_id")
product_category = _Feature("product_category")
advertiser_id = _Feature("advertiser_id")
industry = _Feature("industry")
age_stat1 = _Feature("age_stat1")
age_stat2 = _Feature("age_stat2")
age_stat3 = _Feature("age_stat3")
age_stat4 = _Feature("age_stat4")
age_stat5 = _Feature("age_stat5")
age_stat6 = _Feature("age_stat6")
age_stat7 = _Feature("age_stat7")
age_stat8 = _Feature("age_stat8")
age_stat9 = _Feature("age_stat9")
age_stat10 = _Feature("age_stat10")
gender_stat1 = _Feature("gender_stat1")
gender_stat2 = _Feature("gender_stat2")
pv = _Feature("pv")
gender_ratio1 = _Feature("gender_ratio1", tf.float32, tf.float32)
gender_ratio2 = _Feature("gender_ratio2", tf.float32, tf.float32)
age_ratio1 = _Feature("age_ratio1", tf.float32, tf.float32)
age_ratio2 = _Feature("age_ratio2", tf.float32, tf.float32)
age_ratio3 = _Feature("age_ratio3", tf.float32, tf.float32)
age_ratio4 = _Feature("age_ratio4", tf.float32, tf.float32)
age_ratio5 = _Feature("age_ratio5", tf.float32, tf.float32)
age_ratio6 = _Feature("age_ratio6", tf.float32, tf.float32)
age_ratio7 = _Feature("age_ratio7", tf.float32, tf.float32)
age_ratio8 = _Feature("age_ratio8", tf.float32, tf.float32)
age_ratio9 = _Feature("age_ratio9", tf.float32, tf.float32)
age_ratio10 = _Feature("age_ratio10", tf.float32, tf.float32)

feature_set = [
    rcid_1,
    uid,
    age,
    gender,
    time,
    cid,
    aid,
    product_id,
    product_category,
    advertiser_id,
    industry,
    age_ratio1,
    age_ratio2,
    age_ratio3,
    age_ratio4,
    age_ratio5,
    age_ratio6,
    age_ratio7,
    age_ratio8,
    age_ratio9,
    age_ratio10,
    age_stat1,
    age_stat2,
    age_stat3,
    age_stat4,
    age_stat5,
    age_stat6,
    age_stat7,
    age_stat8,
    age_stat9,
    age_stat10,
    gender_ratio1,
    gender_ratio2,
    gender_stat1,
    gender_stat2,
    pv
    ]

feature_set_v2 = [
    rcid_5,
    uid,
    age,
    gender,
    time,
    cid,
    aid,
    product_id,
    product_category,
    advertiser_id,
    industry,
    age_ratio1,
    age_ratio2,
    age_ratio3,
    age_ratio4,
    age_ratio5,
    age_ratio6,
    age_ratio7,
    age_ratio8,
    age_ratio9,
    age_ratio10,
    age_stat1,
    age_stat2,
    age_stat3,
    age_stat4,
    age_stat5,
    age_stat6,
    age_stat7,
    age_stat8,
    age_stat9,
    age_stat10,
    gender_ratio1,
    gender_ratio2,
    gender_stat1,
    gender_stat2,
    pv
    ]

feature_set_v3 = [
    rcid_10,
    uid,
    age,
    gender,
    time,
    cid,
    aid,
    product_id,
    product_category,
    advertiser_id,
    industry,
    age_ratio1,
    age_ratio2,
    age_ratio3,
    age_ratio4,
    age_ratio5,
    age_ratio6,
    age_ratio7,
    age_ratio8,
    age_ratio9,
    age_ratio10,
    age_stat1,
    age_stat2,
    age_stat3,
    age_stat4,
    age_stat5,
    age_stat6,
    age_stat7,
    age_stat8,
    age_stat9,
    age_stat10,
    gender_ratio1,
    gender_ratio2,
    gender_stat1,
    gender_stat2,
    pv
    ]

feature_set_second_phase = [
    age,
    gender,
    uid,
    rcid_second_phase
]