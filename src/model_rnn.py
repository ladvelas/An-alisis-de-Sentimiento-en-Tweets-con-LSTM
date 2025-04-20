# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def creacion_model_rnn(vocab_size, max_length):
    """
    Crea un modelo RNN simple.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Para clasificación binaria
    ])
    return model