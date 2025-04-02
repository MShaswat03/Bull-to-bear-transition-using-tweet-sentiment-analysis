import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Load predictions
df = pd.read_csv("results/predictions.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# Convert to int if not already
df["Actual"] = df["Actual"].astype(int)
df["Predicted"] = df["Predicted"].astype(int)

# Calculate F1 score
f1 = f1_score(df["Actual"], df["Predicted"], average="weighted")

# Bull/Bear Percentage
bull_count = (df["Predicted"] == 1).sum()
bear_count = (df["Predicted"] == 0).sum()
total = len(df)
bull_percent = 100 * bull_count / total
bear_percent = 100 * bear_count / total

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Actual Price", color="royalblue", linewidth=2)
plt.plot(df["Date"], df["Open"], label="Predicted Price", color="darkorange", linewidth=2)
plt.axvline(x=df["Date"].iloc[int(len(df)*0.7)], color='red', linestyle='--')

# Annotations
plt.title("📈 Price Prediction vs Actual (All Period)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

# Display bull/bear & F1 Score
textstr = f"🐂 Bull: {bull_percent:.2f}%\n🐻 Bear: {bear_percent:.2f}%\n🎯 F1 Score: {f1:.4f}"
plt.gcf().text(0.15, 0.75, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# Save or show
plt.tight_layout()
plt.savefig("results/predicted_vs_actual.png")
plt.show()
