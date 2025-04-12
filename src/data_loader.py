import pandas as pd
import tensorflow as tf

def load_data():
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    csv_path = tf.keras.utils.get_file("twitter_sentiment.csv", url)
    df = pd.read_csv(csv_path)
    return df[['tweet', 'label']]

