import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def compute_tweet_sentiment(text):
    return sia.polarity_scores(text)["compound"]

def process_tweets(input_filepath, output_filepath, text_column="tweet"):
    """
    Processes the tweets CSV.
    - input_filepath: path to the tweets CSV.
    - output_filepath: where to save daily aggregated tweet sentiment.
    - text_column: the name of the column containing tweet text.
      Change this parameter if your CSV uses a different column name.
    """
    df = pd.read_csv(input_filepath)
    # Use the provided text_column. Change "Tweet" if your column is named differently.
    try:
        df["Cleaned_Tweet"] = df[text_column].apply(clean_tweet)
    except KeyError:
        raise KeyError(f"The specified column '{text_column}' was not found in the tweets CSV. Please check your CSV file.")
    df["Tweet_Sentiment"] = df["Cleaned_Tweet"].apply(compute_tweet_sentiment)
    
    # Convert date column; assuming the date column is named "Date". Change if necessary.
    df["date"] = pd.to_datetime(df["date"])
    
    # Aggregate by date (average sentiment)
    daily_tweet = df.groupby(df["date"].dt.date)["Tweet_Sentiment"].mean().reset_index()
    daily_tweet.columns = ["date", "Avg_Tweet_Sentiment"]
    daily_tweet["date"] = pd.to_datetime(daily_tweet["date"])
    
    daily_tweet.to_csv(output_filepath, index=False)
    print(f"Daily tweet sentiment saved to {output_filepath}")
    return daily_tweet

if __name__ == "__main__":
    # Update text_column if needed (e.g., "text" if your column is named that way)
    process_tweets("data/tsla-tweets.csv", "results/daily_tweet_sentiment.csv", text_column="Tweet")
