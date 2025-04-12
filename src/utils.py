import re

def clean_text(text):
    text = re.sub(r'@\w+|#\w+|http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.lower().strip()

