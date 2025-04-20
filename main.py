from merge_all_script import merge_all
from eda_analysis import run_feature_engineering
from train import train_lstm_model
from process_news import process_news
from process_tweets import process_tweets
from merge_stocks import merge_stock_data
from classic_models import train_classic_model
import json
from classic_models import train_classic_model
import json

feature_cols = ["open", "close", "Avg_Tweet_Sentiment", "Avg_News_Sentiment", "value",
                "Prev_Close", "Prev_TweetSentiment", "Prev_NewsSentiment", "EMA_5",
                "EMA_10", "BB_MA", "BB_Upper", "BB_Lower", "RSI", "MACD", "Signal_Line",
                "Lag_Tweet_Sentiment_1", "Lag_News_Sentiment_1"
            ]

def main(): 
    
    model_scores = {} 
    print("🚀 Processing tweet sentiment...")
    process_tweets("data/tsla-tweet.csv", "results/daily_tweet_sentiment.csv")
  
    print("📊 Merging all datasets...")
    merge_all()
  
    print("⚡ Enhancing features with EDA...")
    run_feature_engineering()

    print("🤖 Training LSTM classification model...")
    model, f1_score, accuracy = train_lstm_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=feature_cols, 
        label_col="Label"
        sequence_length=30, 
        model_save_path="results/best_model.pth" 
   ) 

    print(f"🎉 Trained LSTM classification model with F1 Score: {f1_score:.4f} and Accuracy: {accuracy:.4f}")
    model_scores["LSTM"] {"F1": f1_score, "Accuracy": accuracy} 

    print("🌲 Training Random Forest...")
    rf_f1, rf_acc = train_classic_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=feature_cols, 
        label_col="Label"
        model_type="random_forest"
   ) 
   model_scores["Random Forest"] {"F1": rf_f1, "Accuracy": rf_acc} 

   print("🌟 Training Gradient Boost...")
   gb_f1, gb_acc = train_classic_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=feature_cols, 
        label_col="Label"
        model_type="gradient_boost"
   )
   model_scores["Gradient Boost"] {"F1": gb_f1, "Accuracy": gb_acc} 

   print("Training BiLSTM model...")
   model, f1_bilstm, acc_bilstm = train_lstm_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=feature_cols,
        label_col="Label",
        model_type="bilstm",
        model_save_path="results/best_model_bilstm.pth"
    )
    model_scores["BiLSTM"] = {"F1": f1_bilstm, "Accuracy": acc_bilstm}

    print("Training Seq2Seq model...")
    model, f1_s2s, acc_s2s = train_lstm_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=feature_cols,
        label_col="Label",
        model_type="seq2seq",
        model_save_path="results/best_model_seq2seq.pth"
    )
    model_scores["Seq2Seq"] = {"F1": f1_s2s, "Accuracy": acc_s2s}

    print("Training Seq2Seq with BiLSTM encoder...")
    model, f1_s2s_bi, acc_s2s_bi = train_lstm_model(
        csv_file="results/merged_all_data_enriched.csv",
        feature_cols=feature_cols,
        label_col="Label",
        model_type="seq2seq_bilstm",
        model_save_path="results/best_model_seq2seq_bilstm.pth"
    )
    model_scores["Seq2Seq_BiLSTM"] = {"F1": f1_s2s_bi, "Accuracy": acc_s2s_bi}


    # 🔽 Save all scores to JSON
    with open("results/model_scores.json", "w") as f:
        json.dump(model_scores, f, indent=4)
    print("📦 Saved model comparison scores to results/model_scores.json ✅")


if __name__ == "__main__":
    main()
