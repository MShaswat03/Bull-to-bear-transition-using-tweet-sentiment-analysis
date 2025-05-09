import pandas as pd
import os


def merge_all():
    stocks = pd.read_csv("results/merged_stocks.csv")
    tweet_sent = pd.read_csv("results/daily_tweet_sentiment.csv")
    news_sent = pd.read_csv("results/daily_news_sentiment.csv")
    inflation = pd.read_csv("data/US_inflation_rates.csv")

    
    stocks['date'] = pd.to_datetime(stocks['date'], errors='coerce')
    tweet_sent['date'] = pd.to_datetime(tweet_sent['date'], errors='coerce')
    news_sent['date'] = pd.to_datetime(news_sent['date'], errors='coerce')
    inflation['date'] = pd.to_datetime(inflation['date'], errors='coerce')


    stocks.dropna(subset=['date'], inplace=True)
    tweet_sent.dropna(subset=['date'], inplace=True)
    news_sent.dropna(subset=['date'], inplace=True)
    inflation.dropna(subset=['date'], inplace=True)


    merged = pd.merge(stocks, tweet_sent, on="date", how="left")
    merged = pd.merge(merged, news_sent, on="date", how="left")
    merged = pd.merge(merged, inflation, on="date", how="left")
    
    
    merged["Avg_Tweet_Sentiment"] = merged["Avg_Tweet_Sentiment"].fillna(0)
    merged["Avg_News_Sentiment"] = merged["Avg_News_Sentiment"].fillna(0)
    merged["value"] = merged["value"].ffill().fillna(0)


    merged["Prev_Close"] = merged["close"].shift(1)
    merged["Prev_TweetSentiment"] = merged["Avg_Tweet_Sentiment"].shift(1)
    merged["Prev_NewsSentiment"] = merged["Avg_News_Sentiment"].shift(1)
    merged = merged.dropna(subset=["Prev_Close", "Prev_TweetSentiment", "Prev_NewsSentiment"])


    merged["Label"] = (merged["open"] > merged["Prev_Close"]).astype(int)

    os.makedirs("results", exist_ok=True)
    merged.to_csv("results/merged_all_data_classification.csv", index=False)
    print("✅ Merged all data saved to results/merged_all_data_classification.csv")


if __name__ == "__main__":
    merge_all()
