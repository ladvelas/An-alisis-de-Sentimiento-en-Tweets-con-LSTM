import re

def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # eliminar menciones
    text = re.sub(r'#\w+', '', text)  # eliminar hashtags
    text = re.sub(r'http\S+', '', text)  # eliminar URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # eliminar caracteres especiales
    text = re.sub(r'\s+', ' ', text)  # eliminar espacios múltiples
    return text.strip()