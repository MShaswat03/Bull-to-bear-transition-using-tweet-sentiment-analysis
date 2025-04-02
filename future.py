import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import matplotlib.pyplot as plt

# Number of days to predict into the future
n_days = 730  # ~2 years

# Load dataset
df = pd.read_csv("data/tesla_stocks.csv")
df.rename(columns=str.lower, inplace=True)
df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace=True)

# Prepare training data
df["prev_close"] = df["close"].shift(1)
df.dropna(inplace=True)

X = df[["prev_close"]].values
y = df["open"].values

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Train the model
model = LinearRegression()
model.fit(X_scaled, y_scaled)

# Start prediction using the last available close price
latest_close = df["close"].iloc[-1]
last_date = df["date"].iloc[-1]

future_dates = []
future_predictions = []

for i in range(n_days):
    # Scale and predict
    latest_scaled = scaler_X.transform([[latest_close]])
    pred_scaled = model.predict(latest_scaled)
    pred_open = scaler_y.inverse_transform(pred_scaled)[0][0]

    # Save prediction and date
    future_dates.append(last_date + timedelta(days=i + 1))
    future_predictions.append(pred_open)

    # Set up next input
    latest_close = pred_open  # Assume predicted open becomes next close (simplified)

# Create prediction DataFrame
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Open": future_predictions
})

# Output preview
print(f"📅 Latest known date: {last_date.date()}")
print(f"📆 Predicting for next {n_days} days until: {future_df['Date'].iloc[-1].date()}")
print(f"🔮 First predicted open: ${future_df['Predicted_Open'].iloc[0]:.2f}")
print(f"🔮 Last predicted open: ${future_df['Predicted_Open'].iloc[-1]:.2f}")

# Optional: Save to CSV
future_df.to_csv("results/future_open_predictions.csv", index=False)
print("✅ Saved to results/future_open_predictions.csv")

# Optional: Plot the predictions
plt.figure(figsize=(12, 5))
plt.plot(future_df["Date"], future_df["Predicted_Open"], label="Predicted Open", color="orange")
plt.title("🔮 Future Tesla Opening Prices (Next 2 Years)")
plt.xlabel("Date")
plt.ylabel("Predicted Open Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/future_open_plot.png")
plt.show()
