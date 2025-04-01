import pandas as pd
import os
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already available.
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    """Clean text by removing URLs, mentions, and special characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def compute_news_sentiment(text):
    """Compute the compound sentiment score using VADER."""
    return sia.polarity_scores(text)["compound"]

def process_news(input_filepath, output_filepath, text_column="content"):
    """
    Process news data:
      - Clean the text from the specified text_column.
      - Compute sentiment scores.
      - Aggregate daily average sentiment.
      - Save the daily news sentiment to a CSV.
    """
    df = pd.read_csv(input_filepath)
    # Clean news text and compute sentiment.
    df["Cleaned_Text"] = df[text_column].apply(clean_text)
    df["News_Sentiment"] = df["Cleaned_Text"].apply(compute_news_sentiment)
    # Convert the date column to datetime (assumes the column is named 'date').
    df["date"] = pd.to_datetime(df["date"])
    # Group by date (as date only) and compute the daily average sentiment.
    daily_news = df.groupby(df["date"].dt.date)["News_Sentiment"].mean().reset_index()
    daily_news.columns = ["date", "Avg_News_Sentiment"]
    daily_news["date"] = pd.to_datetime(daily_news["date"])
    daily_news.to_csv(output_filepath, index=False)
    print(f"Daily news sentiment saved to {output_filepath}")
    return daily_news

def merge_all():
    """
    Merge all datasets into one:
      - Loads merged stock data (e.g., from Yahoo Finance and your own stocks file).
      - Loads daily tweet sentiment.
      - Processes news data (using process_news) from your provided news.csv.
      - Loads US inflation data (assumes inflation values are in the 'value' column).
      - Merges them on date (using left joins on stock data so that you keep the full stock date range).
      - Fills missing tweet/news sentiment with 0 and fills inflation via forward-fill.
      - Shifts previous day's features and creates a classification label:
          Label = 1 if today's Open > previous day's Close, else 0.
      - Saves the final merged dataset.
    """
    # Load stock data
    stocks = pd.read_csv("results/merged_stocks.csv")
    # Load daily tweet sentiment
    tweet_sent = pd.read_csv("results/daily_tweet_sentiment.csv")
    # Process news sentiment from the provided news.csv file.
    news_sent = process_news("data/news.csv", "results/daily_news_sentiment.csv", text_column="content")
    # Load US inflation data (assumes it has columns: date, value)
    inflation = pd.read_csv("data/US_inflation_rates.csv")
    
    # Convert date columns to datetime.
    stocks['date'] = pd.to_datetime(stocks['date'])
    tweet_sent['date'] = pd.to_datetime(tweet_sent['date'])
    news_sent['date'] = pd.to_datetime(news_sent['date'])
    inflation['date'] = pd.to_datetime(inflation['date'])
    
    # Merge using left joins on stocks (assumes stocks cover the full period, e.g. 2010-2020)
    merged = pd.merge(stocks, tweet_sent, on="date", how="left")
    merged = pd.merge(merged, news_sent, on="date", how="left")
    merged = pd.merge(merged, inflation, on="date", how="left")
    
    # Fill missing sentiment values with 0; fill missing inflation using forward-fill.
    merged["Avg_Tweet_Sentiment"] = merged["Avg_Tweet_Sentiment"].fillna(0)
    merged["Avg_News_Sentiment"] = merged["Avg_News_Sentiment"].fillna(0)
    merged["value"] = merged["value"].ffill().fillna(0)
    
    merged = merged.sort_values("date").reset_index(drop=True)
    
    # Shift previous day's features.
    merged["Prev_Close"] = merged["Close"].shift(1)
    merged["Prev_TweetSentiment"] = merged["Avg_Tweet_Sentiment"].shift(1)
    merged["Prev_NewsSentiment"] = merged["Avg_News_Sentiment"].shift(1)
    merged = merged.dropna(subset=["Prev_Close", "Prev_TweetSentiment", "Prev_NewsSentiment"])
    
    # Create classification label: 1 if today's Open > previous day's Close, else 0.
    merged["Label"] = (merged["Open"] > merged["Prev_Close"]).astype(int)
    
    os.makedirs("results", exist_ok=True)
    merged.to_csv("results/merged_all_data_classification.csv", index=False)
    print("Merged all data saved to results/merged_all_data_classification.csv")
    return merged

if __name__ == "__main__":
    merge_all()
