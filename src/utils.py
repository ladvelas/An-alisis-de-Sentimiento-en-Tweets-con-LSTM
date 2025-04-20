import pandas as pd
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split


def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # eliminar menciones
    text = re.sub(r'#\w+', '', text)  # eliminar hashtags
    text = re.sub(r'http\S+', '', text)  # eliminar URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # eliminar caracteres especiales
    text = re.sub(r'\s+', ' ', text)  # eliminar espacios múltiples
    text = text.lower()  # convertir a minúsculas
    text = re.sub(r'\d+', '', text)  # eliminar números
    text = re.sub(r'\s+', ' ', text)  # eliminar espacios múltiples
    return text.strip()

# Función para equilibrar el dataset
def balance_dataset(df):
    # Contar la cantidad de muestras por clase
    counts = df['label'].value_counts()
    
    # Encontrar la clase mayoritaria
    major_class = counts.idxmax()
    
    # Encontrar la cantidad de muestras de la clase mayoritaria
    major_count = counts.max()
    
    # Crear un nuevo dataframe vacío para almacenar las muestras equilibradas
    balanced_df = pd.DataFrame(columns=df.columns)
    
    # Iterar sobre cada clase y agregar muestras al nuevo dataframe
    for label, count in counts.items():
        if label == major_class:
            balanced_df = pd.concat([balanced_df, df[df['label'] == label]])
        else:
            # Calcular el número de muestras a agregar para equilibrar
            num_samples_to_add = major_count - count
            
            # Seleccionar aleatoriamente muestras de la clase minoritaria
            samples_to_add = df[df['label'] == label].sample(num_samples_to_add, replace=True)
            
            # Agregar las muestras al nuevo dataframe
            balanced_df = pd.concat([balanced_df, samples_to_add])
    
    return balanced_df.reset_index(drop=True)


# Función para tokenizar el texto
def tokenize_text(text_df, num_words=10000, oov_token="<OOV>", max_len=50):
    # Tokenización
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(text_df)

    sequences = tokenizer.texts_to_sequences(text_df)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded

def binary_labels(df):
    df['label'] = df['label'].astype('int32')
    return df

def get_train_test_validation(X, y):
    # Dividir entre train y temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Dividir X_temp entre validación y test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
