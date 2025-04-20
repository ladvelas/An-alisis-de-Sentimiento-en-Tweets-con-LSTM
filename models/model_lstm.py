from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Ejemplo de datos
texts = ["Este es un tweet", "Otro tweet aquí"]
labels = np.array([1, 0])  # Convertido a array

# Tokenización
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
X_train = tokenizer.texts_to_sequences(texts)

# Padding
X_train_padded = pad_sequences(X_train, maxlen=100)

# Verifica forma
print(f"Shape de X_train_padded: {X_train_padded.shape}")

# Modelo
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compilación
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(X_train_padded, labels, epochs=5, batch_size=2)
