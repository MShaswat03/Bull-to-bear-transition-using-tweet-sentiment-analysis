from merge_all_script import merge_all
from eda_analysis import run_feature_engineering
from train import train_lstm_model
from process_news import process_news
from process_tweets_fin import process_tweets
from merge_stocks import merge_stock_data

def main():
    print("🚀 Processing tweet sentiment...")
    process_tweets("data/tsla-tweets.csv", "results/daily_tweet_sentiment.csv")

    print("📈 Merging stock data...")
    # Merge multiple stocks into a single dataset
    stocks = merge_stock_data(
        [
            "data/tesla_stocks.csv",
            "data/AAPL.csv",  # ❌ Incorrect here!
            "data/GOOGL.csv",
            "data/META.csv"
        ],
        "data/Yahoo_Tesla.csv"
    )
    stocks.to_csv("results/merged_stocks.csv", index=False)

    #testing
    print("📊 Merging all datasets...")
    merge_all()

    print("⚡ Enhancing features with EDA...")
    run_feature_engineering()

    print("🤖 Training LSTM classification model...")
    model, f1_score, accuracy = train_lstm_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=[
            "open", "close", "Avg_Tweet_Sentiment", "Avg_News_Sentiment", "value",
            "Prev_Close", "Prev_TweetSentiment", "Prev_NewsSentiment", "EMA_5",
            "EMA_10", "BB_MA", "BB_Upper", "BB_Lower", "RSI", "MACD", "Signal_Line",
            "Lag_Tweet_Sentiment_1", "Lag_News_Sentiment_1"
        ],
        label_col="Label",
        sequence_length=30,
        model_save_path="results/best_model.pth"
    )
    print(f"🎉 Trained LSTM classification model with F1 Score: {f1_score:.4f} and Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
