import tensorflow as tf
from tensorflow.keras.layers import Dropout, SpatialDropout2D
import random

class MixedDropout:
    def __init__(self, spatial_dropout_dist=0.2, dropout_dist=0.4):
        self.dropout = Dropout(dropout_dist)
        self.dropout_bi = SpatialDropout2D(spatial_dropout_dist)
    
    def __call__(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout_bi(x)