import numpy as np

from src.data_loader import load_data
from src.utils import (
    clean_df,
    balance_dataset,
    tokenize_text,
    binary_labels,
    split_data,
)

from src.model_rnn import creacion_model_rnn
from src.train import copilacion_train
from src.evaluate import (
    evaluate_model, 
    guardar_modelo,
)

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
model_path = "../models/rnn_model.keras"

# Carga, limpieza y balanceo de datos
df = load_data(url)
df_clean = clean_df(df)
df_balanced = balance_dataset(df_clean)
df_balanced = binary_labels(df_balanced)

# Tokenización y creación de etiquetas binarias
X = tokenize_text(df_balanced["tweet"])
y = np.array(df_balanced["label"])

# División de datos en train, val y test
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Creación del modelo
model = creacion_model_rnn()

# Compilación y entrenamiento del modelo
history = copilacion_train(model, X_train, y_train, X_val, y_val, X_test, y_test)

# Evaluación del modelo
loss, accuracy = evaluate_model(model, X_test, y_test)

#Guardar el modelo
guardar_modelo(model, model_path)


