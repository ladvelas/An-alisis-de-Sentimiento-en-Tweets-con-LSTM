from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# 🔹 Datos de ejemplo
texts = ["Este es un tweet", "Otro tweet aquí"]
labels = [1, 0]

# 🔹 Tokenización
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
X_train = tokenizer.texts_to_sequences(texts)

# 🔹 Padding
X_train_padded = pad_sequences(X_train, maxlen=100)

# 🔹 Conversión a np.array
X_train_padded = np.array(X_train_padded)
labels = np.array(labels)

# 🔹 Definición del modelo RNN
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(SimpleRNN(128))  # Aquí está el RNN
model.add(Dense(1, activation='sigmoid'))

# 🔹 Compilar modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 Entrenamiento
model.fit(X_train_padded, labels, epochs=5, batch_size=32)


