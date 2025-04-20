import pandas as pd
import numpy as np


def add_technical_indicators(df): 
    df['EMA_5'] = df['Prev_Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Prev_Close'].ewm(span=10, adjust=False).mean()

    # Bollinger Bands 
    df['BB_MA'] = df['Prev_Close'].rolling(window=20).mean()
    df['BB_Upper'] - df['BB_MA'] + 2 * df['Prev_Close'].rolling(window=20).std()
    df['BB_Lower'] - df['BB_MA'] + 2 * df['Prev_Close'].rolling(window=20).std()

    # RSI Calculation 
    delta = df['Prev_Close'],diff()
    gain = delta.where(delta > 0, 0) 
    loss = -delta.where(delta < 0, 0) 
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain /  avg_loss 
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['Prev_Close'].ewm(span=12, adjust=False).mean() -  df['Prev_Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean() 

    # Lagged Sentiments 
    df['Lag_Tweet_Sentiment_1'] = df['Prev_TweetSentiment'].shift(1)
    df['Lag_News_Sentiment_1'] = df['Prev_NewsSentiment'].shift(1)

    df.dropna(inplace=True)
    return df

def run_feature_engineering(): 
    df = pd.read_csv("results/merdged_all_data_classification.csv")
    df['date'] = pd.to_datetime(df['date']) 
    df = df.sort_values("date").reset_index(drop=True)

    df = add_technical_indicators(df)

    df.to_csv("results/merged_all_data_enriched.csv", index=False)

if __name__ == "__main__": 
    run_feature_engineering()
