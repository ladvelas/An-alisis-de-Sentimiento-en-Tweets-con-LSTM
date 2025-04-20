import pandas as pd
import tensorflow as tf

from src.utils import clean_text

def load_data():
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    #csv_path = tf.keras.utils.get_file("twitter_sentiment.csv", url)
    df = pd.read_csv(url)
    return df[['tweet', 'label']]

df_x = load_data()
df_x['cleaned_tweet'] = df_x['tweet'].apply(clean_text)

df_x[['tweet', 'cleaned_tweet', 'label']].head()

