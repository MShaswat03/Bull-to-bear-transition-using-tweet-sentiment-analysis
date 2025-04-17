from merge_all_script import merge_all
from eda_analysis import run_feature_engineering
from train import train_lstm_model
from process_news import process_news
from process_tweets import process_tweets
from merge_stocks import merge_stock_data
from classic_models import train_classic_model
import json

def main():
    # Dictionary to store model scores
    model_scores = {}

    print("🚀 Processing tweet sentiment...")
    process_tweets("data/tsla-tweets.csv", "results/daily_tweet_sentiment.csv")

    print("📈 Merging stock data...")
    stocks = merge_stock_data(
        [
            "data/tesla_stocks.csv",
            "data/AAPL.csv",
            "data/GOOGL.csv",
            "data/META.csv"
        ],
        "data/Yahoo_Tesla.csv"
    )
    stocks.to_csv("results/merged_stocks.csv", index=False)

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
    model_scores["LSTM"] = {"F1": f1_score, "Accuracy": accuracy}

    print("🌲 Training Random Forest...")
    rf_f1, rf_acc = train_classic_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=[
            "open", "close", "Avg_Tweet_Sentiment", "Avg_News_Sentiment", "value",
            "Prev_Close", "Prev_TweetSentiment", "Prev_NewsSentiment", "EMA_5",
            "EMA_10", "BB_MA", "BB_Upper", "BB_Lower", "RSI", "MACD", "Signal_Line",
            "Lag_Tweet_Sentiment_1", "Lag_News_Sentiment_1"
        ],
        label_col="Label",
        model_type="random_forest"
    )
    model_scores["Random Forest"] = {"F1": rf_f1, "Accuracy": rf_acc}

    print("🌟 Training Gradient Boost...")
    gb_f1, gb_acc = train_classic_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=[
            "open", "close", "Avg_Tweet_Sentiment", "Avg_News_Sentiment", "value",
            "Prev_Close", "Prev_TweetSentiment", "Prev_NewsSentiment", "EMA_5",
            "EMA_10", "BB_MA", "BB_Upper", "BB_Lower", "RSI", "MACD", "Signal_Line",
            "Lag_Tweet_Sentiment_1", "Lag_News_Sentiment_1"
        ],
        label_col="Label",
        model_type="gradient_boost"
    )
    model_scores["Gradient Boost"] = {"F1": gb_f1, "Accuracy": gb_acc}

    # 🔽 Save all scores to JSON
    with open("results/model_scores.json", "w") as f:
        json.dump(model_scores, f, indent=4)
    print("📦 Saved model comparison scores to results/model_scores.json ✅")


if __name__ == "__main__":
    main()
