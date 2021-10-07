import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Input, MaxPooling2D, BatchNormalization, Dropout, Reshape, Permute, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def conv_block(input,filters, kernel, pad, stride):
    
    x = Conv2D(filters, kernel_size=kernel, padding=pad, activation='relu')(input)
    x = Conv2D(filters, kernel_size=kernel, padding=pad, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=kernel, padding=pad, strides=stride, activation='relu')(x)
    
    return x

def dsc_block(input, kernel, pad, stride):

    x = DepthwiseConv2D(kernel_size=kernel, padding=pad, strides=stride, activation='relu')(input)
    x = Dropout(0.2)(x)
    x = DepthwiseConv2D(kernel_size=kernel, padding=pad, strides=stride, activation='relu')(x) 
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=kernel, padding=pad, strides=stride, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Add()([input, x])
    return x



def get_model(input_shape, out_tokens):
    
    input = Input(shape=input_shape, name='the_input')

    ### CB1
    
    x = conv_block(input, 32, (3,3), "same", (1,1))
    #x = MaxPooling2D()(x)

    ### CB2
    
    x = conv_block(x, 64, (3,3), "same", (2,2))

    ### CB3
    
    x = conv_block(x, 128, (3,3), "same", (2,2))

    ### CB4

    x = conv_block(x, 256, (3,3), "same", (2,2))

    ### CB5

    x = conv_block(x, 512, (3,3), "same", (2,1))

    ### CB6
    
    x = conv_block(x, 512, (3,3), "same", (2,1))

    ### DSCB_Place

    x = dsc_block(x, (3,3), "same", (1,1))
    x = dsc_block(x, (3,3), "same", (1,1))
    x = dsc_block(x, (3,3), "same", (1,1))
    x = dsc_block(x, (3,3), "same", (1,1))

    x = Conv2D(out_tokens+1, kernel_size=(5,5), padding="same", activation="softmax")(x)

    x = Permute((2, 1, 3))(x)
    y_pred = Reshape(target_shape=(-1, out_tokens+1), name='reshape')(x)

    model_pr = Model(inputs=input, outputs=y_pred)
    model_pr.summary()

    labels = Input(name='the_labels',shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model_tr = Model(inputs=[input, labels, input_length, label_length],
                  outputs=loss_out)

    model_tr.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    return model_tr, model_pr
