import pandas as pd
from datetime import datetime

# Load predictions and process dates
df = pd.read_csv("results/predictions.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Fill missing dates using interpolation (business days)
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')  # Business days only
df = df.reindex(full_index)

# Interpolate numeric columns
df["Open"] = df["Open"].interpolate()
df["Close"] = df["Close"].interpolate()
df["Predicted"] = df["Predicted"].interpolate().round().fillna(0).astype(int)
df["Actual"] = df["Actual"].interpolate().round().fillna(0).astype(int)

# Fill company name (forward fill)
df["Company"] = df["Company"].ffill()

# Reset index to make Date a column again
df.reset_index(inplace=True)
df.rename(columns={"index": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"]).dt.date

# 🧠 Input date
input_str = input("📅 Enter the date (YYYY-MM-DD): ")
try:
    input_date = datetime.strptime(input_str, "%Y-%m-%d").date()
except ValueError:
    print("❌ Invalid date format. Please use YYYY-MM-DD.")
    exit()

# 🔍 Try to find exact prediction
if input_date in df["Date"].values:
    row = df[df["Date"] == input_date].iloc[0]
else:
    # Find nearest available date
    all_dates = df["Date"].sort_values()
    closest_date = min(all_dates, key=lambda d: abs(d - input_date))
    row = df[df["Date"] == closest_date].iloc[0]
    print(f"⚠️ No prediction for {input_date}. Using closest date: {closest_date}")

# 🧾 Show output
print("\n🔎 Prediction Details")
print(f"📅 Date: {row['Date']}")
print(f"🏢 Company: {row['Company']}")
print(f"💰 Predicted Opening Price: {row['Open']:.2f}")
print(f"📉 Closing Price: {row['Close']:.2f}")
print(f"📈 Market Trend: {'Bull 🐂' if row['Predicted'] == 1 else 'Bear 🐻'}")
