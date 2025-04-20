import pandas as pd
# url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"

def load_data(url):
    df = pd.read_csv(url)
    return df