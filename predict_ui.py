import streamlit as st
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(page_title="📊 Tesla Market Prediction", layout="centered")
st.markdown("<h1 style='text-align: center; font-size: 48px;'>📊 Tesla Stock Market Trend Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 20px;'>Enter a date to get the predicted market trend and opening price</p>", unsafe_allow_html=True)

# Load and preprocess the dataset
df = pd.read_csv("results/predictions.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Fill all business days
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
df = df.reindex(full_index)

# Interpolation and cleanup
df["Open"] = df["Open"].interpolate()
df["Close"] = df["Close"].interpolate()
df["Predicted"] = df["Predicted"].interpolate().round().fillna(0).astype(int)
df["Actual"] = df["Actual"].interpolate().round().fillna(0).astype(int)
df["Company"] = df["Company"].ffill()
df.reset_index(inplace=True)
df.rename(columns={"index": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"]).dt.date

# User input
input_date = st.date_input("📅 **Select a date:**", value=datetime.today().date())

# Find prediction row
if input_date in df["Date"].values:
    row = df[df["Date"] == input_date].iloc[0]
    st.success("✅ Exact match found for your input.")
else:
    all_dates = df["Date"].sort_values()
    closest_date = min(all_dates, key=lambda d: abs(d - input_date))
    row = df[df["Date"] == closest_date].iloc[0]
    st.success("✅ Exact match found for your input.")

# Display results with bigger fonts
st.markdown("### 🔍 Prediction Details", unsafe_allow_html=True)
st.markdown(f"""
<div style='background-color: #1e1e1e; padding: 30px; border-radius: 12px; font-size: 22px; line-height: 2;'>
    <p><strong>📅 Date:</strong> {row['Date']}</p>
    <p><strong>🏢 Company:</strong> {row['Company']}</p>
    <p><strong>💰 Predicted Opening Price:</strong> <span style='color: #00ff99;'>${row['Open']:.2f}</span></p>
    <p><strong>📉 Closing Price:</strong> <span style='color: #ffcc00;'>${row['Close']:.2f}</span></p>
    <p><strong>📈 Market Trend:</strong> {"<span style='color: lightgreen;'>Bull 🐂</span>" if row['Predicted'] == 1 else "<span style='color: tomato;'>Bear 🐻</span>"}</p>
</div>
""", unsafe_allow_html=True)
