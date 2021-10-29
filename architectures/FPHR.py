from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MultiHeadAttention, Embedding, LayerNormalization, Softmax, Conv1D, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.saving.save import save_model
from NetBlocks import PositionEncoding1D, PositionEncoding2D
import numpy as np

def get_encoder(in_tens, d_model, max_height, max_width):

    x = ResNet50(include_top=False, weights=None, input_shape=(None, None, 1))(in_tens)
    x = Conv2D(d_model, kernel_size=(1,1), padding="same")(x) # Convolutional projection
    x = PositionEncoding2D(d_model, max_width, max_height)(x)
    x = Reshape(target_shape=(-1, d_model), name='reshape')(x)

    return x

def get_decoder_layer(in_dec, context_vector, look_ahead_mask, d_model, n_heads, ff_depth, dropout_rate=0.5):
    
    x = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(query=in_dec, value=in_dec, attention_mask = look_ahead_mask)
    x = Dropout(dropout_rate)(x)
    x_ln_1 = LayerNormalization(epsilon=1e-6)(in_dec+x)

    x2 = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(query=context_vector, value=x_ln_1)
    x2 = Dropout(dropout_rate)(x2)
    x_ln_2 = LayerNormalization(epsilon=1e-6)(x_ln_1 + x2)

    x3 = Dense(ff_depth, activation='relu')(x_ln_2)
    x3 = Dense(d_model)(x3)
    output = LayerNormalization(epsilon=1e-6)(x3 + x_ln_2)

    return output

def get_decoder(decoder_layers, decoder_input, context_vector, look_ahead_mask, max_seq_len, d_model, n_heads, ff_depth, vocab_len, dropout=0.5):

    x = Embedding(input_dim=vocab_len, output_dim=d_model)(decoder_input)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x = PositionEncoding1D(d_model,max_seq_len)(x)
    x = Dropout(dropout)(x)

    for _ in range(decoder_layers):
        x = get_decoder_layer(x, context_vector, look_ahead_mask, d_model, n_heads, ff_depth)

    x = Conv1D(vocab_len, kernel_size=1)(x)
    output = Softmax()(x)
    
    return output

def get_model(max_height, max_width ,n_decoder_layers, max_seq_len, n_heads, ff_depth, target_len, d_model):
    input_image = Input(shape=(None,None,1))
    input_decoder = Input(shape=(None, ))
    look_ahead_mask = Input(shape=(None, max_seq_len+1, max_seq_len+1))
    
    encoder_output = get_encoder(input_image, d_model, max_height, max_width)
    out = get_decoder(n_decoder_layers, input_decoder, encoder_output, look_ahead_mask, max_seq_len, d_model, n_heads, ff_depth, target_len)

    model = Model(inputs=[input_image, input_decoder, look_ahead_mask], outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='categorical_crossentropy')
    model.summary()

    return model

def main():
    get_model(256,512, 6, 512, 4, 1024, 360, 260)

if __name__ == "__main__":
    main()
