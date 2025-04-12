import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from torch.nn.functional import softmax

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1).detach().numpy()[0]
    sentiment_score = probs[2] - probs[0]  # positive - negative
    return sentiment_score

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
    df["Tweet_Sentiment"] = df["Cleaned_Tweet"].apply(get_finbert_sentiment)
    
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
    process_tweets("data/tsla-tweets.csv", "results/daily_tweet_sentiment_fin.csv", text_column="Tweet")
