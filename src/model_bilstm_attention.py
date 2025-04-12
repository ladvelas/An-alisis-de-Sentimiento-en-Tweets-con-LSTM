from tensorflow.keras.layers import Layer, Dense, Embedding, Bidirectional, LSTM, Input
from tensorflow.keras.models import Model
import tensorflow as tf

class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def call(self, inputs):
        score = tf.nn.tanh(inputs)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

def build_bilstm_attention(vocab_size, embedding_dim, input_length):
    inputs = Input(shape=(input_length,))
    x = Embedding(vocab_size, embedding_dim, input_length=input_length)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Attention()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)
