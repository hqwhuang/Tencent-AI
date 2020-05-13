import tensorflow as tf
import numpy as np


class _Feature(collections.namedtuple("_Feature", [
    'feature_name', 'input_type', 'output_type'
])):
    def __new__(cls,
                feature_name,
                input_type,
                output_type)
        return super(_Feature, cls).__new__(
            cls,
            feature_name,
            input_type,
            output_type
        )
    
    
    def transform_feature(self, read_data):
        pass
    

    def input_feature(self, read_feature):
        pass