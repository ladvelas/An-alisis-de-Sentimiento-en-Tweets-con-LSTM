from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y):
    # Medición de la precisión en el conjunto de test
    loss, accuracy = model.evaluate(X, y)
    print(f'Accuracy en test: {accuracy:.4f}')
    return loss, accuracy

def guardar_modelo(model, path):
    # Guardar el modelo entrenado
    model.save(path)
    print(f'Modelo guardado en {path}')


