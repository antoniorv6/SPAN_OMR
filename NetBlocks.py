import tensorflow as tf
from tensorflow.keras.layers import Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU
import random
from tensorflow_addons.layers.normalizations import InstanceNormalization

class MixedDropout:
    def __init__(self, spatial_dropout_dist=0.2, dropout_dist=0.4):
        super(MixedDropout, self).__init__()
        self.dropout = Dropout(dropout_dist)
        self.dropout_bi = SpatialDropout2D(spatial_dropout_dist)
    
    def __call__(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout_bi(x)

class ConvBlock():
    
    def __init__(self, filters, kernel, pad, stride):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=kernel, padding=pad)
        self.conv2 = Conv2D(filters, kernel_size=kernel, padding=pad)
        self.conv3 = Conv2D(filters, kernel_size=kernel, strides=stride, padding=pad)
        self.activation = ReLU()
        self.dropout = MixedDropout()
        self.norm = InstanceNormalization()
    
    def __call__(self, input):
        
        pos = random.randint(1,3)

        x = self.conv1(input)
        x = self.activation(x)

        if pos==1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.norm(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x


class DepthSepConvBlock():
    def __init__(self, filters, krn_sz, pad, stride=(1,1)):
        super(DepthSepConvBlock, self).__init__()
        self.depth_conv = DepthwiseConv2D(kernel_size=krn_sz, dilation_rate=(1,1), strides=stride, padding=pad)
        self.point_conv = Conv2D(filters, kernel_size=(1,1), dilation_rate=(1,1))
    
    def __call__(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class DSCBlock():

    def __init__(self, kernel, pad, stride):
        super(DSCBlock, self).__init__()
        self.conv1 = DepthSepConvBlock(filters=512, krn_sz=kernel, pad=pad)
        self.conv2 = DepthSepConvBlock(filters=512, krn_sz=kernel, pad=pad)
        self.conv3 = DepthSepConvBlock(filters=512, krn_sz=kernel, stride=stride, pad=pad)
        self.activation = ReLU()
        self.dropout = MixedDropout()
        self.norm = InstanceNormalization()
    
    def __call__(self, input):
        
        pos = random.randint(1,3)

        x = self.conv1(input)
        x = self.activation(x)

        if pos==1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.norm(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return input + x
        

    

