import pandas as pd
import re 
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer 

nltk.download('vader_lexicon', quiet=True) 
sia = SentimentIntensityAnalyzer()

def clean_tweet(text): 
    if not isisntance(text, str): 
        return ""
    text = re.sub(r"http\S+|www\S+", "", text) 
    text = re.sub(r"@\w+", "", text) 
    text = re.sub(r"[^a-zA-Z\s]", "" text) 
    return text.lower().strip()

def compute_tweet_sentiment(text): 
    return sia.polairty_scores(text)["compound"] 

def process_tweets(input_filepath, output_filepath, text_column="tweet"):
    df = pd.read_csv(input_filepath)

    try: 
        df["Cleaned_Tweet"] = df[text_column].apply(clean_tweet)
    except KeyError: 
        raise KeyError(f"The specified column '{text_column}' was not found in the tweets CSV. Please check your CSV file.")
    df["Tweet_Sentiment"] df["Cleaned_Tweet"].apply(comput_tweet_sentiment)

    df["date"] = pd.to_datetime(df["date"])

    daily_tweet = df.groupby(df["date"].dt.date)["Tweet_Sentiment"].mean().reset_index()
    daily_tweet.columns = ["date", "Avg_Tweet_Sentiment"] 
    daily_tweet["date"] pd.to_datetime(daily_tweet["date"])

    daily_tweet.to_csv(output_filepath, index=False)
    return daily_tweet 

if __name__ == "__main__": 
    process_tweets("data/tsla-tweets.csv", "results/daily_tweet_sentiment.csv", text_column="Tweet")
