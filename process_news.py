import pandas as pd
import os 
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

def process_tweets(input_filepath, output_filepath, text_column="content"):
    df = pd.read_csv(input_filepath)
    df["Cleaned_Tweet"] = df[text_column].apply(clean_tweet)
    df["New_Sentiment"] = df["Cleaned_Text"].apply(comput_new_sentiment) 
    df["date"] = pd.to_datetime(df["date"])
    daily_news = df.groupby(df["date"].dt.date)["News_Sentiment"].mean().reset_index()
    daily_news.columns = ["date", "Avg_News_Sentiment"] 
    daily_news["date"] pd.to_datetime(daily_tweet["date"])
    daily_news.to_csv(output_filepath, index=False)
    return daily_tweet 

def merge_all():

    stocks = pd.read_csv("results/merged_stocks.csv")
    tweet_sent = pd.read_csv("results/daily_tweet_sentiment.csv")
    news_sent = process_news("data/news.csv", "results/daily_news_sentiment.csv", text_column="content")
    inflation = pd.read_csv("data/US_inflation_rates.csv")
    
    stocks['date'] = pd.to_datetime(stocks['date'])
    tweet_sent['date'] = pd.to_datetime(tweet_sent['date'])
    news_sent['date'] = pd.to_datetime(news_sent['date'])
    inflation['date'] = pd.to_datetime(inflation['date'])

    merged = pd.merge(stocks, tweet_sent, on="date", how="left")
    merged = pd.merge(merged, news_sent, on="date", how="left")
    merged = pd.merge(merged, inflation, on="date", how="left")
    

    merged["Avg_Tweet_Sentiment"] = merged["Avg_Tweet_Sentiment"].fillna(0)
    merged["Avg_News_Sentiment"] = merged["Avg_News_Sentiment"].fillna(0)
    merged["value"] = merged["value"].ffill().fillna(0)
    
    merged = merged.sort_values("date").reset_index(drop=True)
    

    merged["Prev_Close"] = merged["Close"].shift(1)
    merged["Prev_TweetSentiment"] = merged["Avg_Tweet_Sentiment"].shift(1)
    merged["Prev_NewsSentiment"] = merged["Avg_News_Sentiment"].shift(1)
    merged = merged.dropna(subset=["Prev_Close", "Prev_TweetSentiment", "Prev_NewsSentiment"])


    merged["Label"] = (merged["Open"] > merged["Prev_Close"]).astype(int)
    
    os.makedirs("results", exist_ok=True)
    merged.to_csv("results/merged_all_data_classification.csv", index=False)
    print("Merged all data saved to results/merged_all_data_classification.csv")
    return merged

if __name__ == "__main__":
    merge_all()
