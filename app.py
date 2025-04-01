import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from model import initialize_model, load_model
import asyncio
import sys

if sys.platform == "darwin" and hasattr(asyncio, "set_event_loop_policy"):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


# --- Load Best Model ---
@st.cache_resource
def load_best_model(model_path="results/best_model.pth", input_size=18, num_classes=2):
    model = initialize_model(input_size=input_size, num_classes=num_classes)
    load_model(model_path, model)
    model.eval()
    return model


# --- Prediction Function ---
def predict_future(model, input_data, num_days=365):
    model.eval()
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_days):
            prediction = model(input_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
            predictions.append(predicted_class)
            
            # Shift input data and append the prediction
            new_input = np.roll(input_data, -1)
            new_input[-1] = predicted_class
            input_tensor = torch.tensor(new_input, dtype=torch.float32).unsqueeze(0)

    return predictions


# --- Generate Future Dates ---
def generate_future_dates(start_date, num_days):
    return [start_date + timedelta(days=i) for i in range(num_days)]


# --- Plot Predictions ---
def plot_predictions(dates, predicted_prices, title="Stock Price Prediction"):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predicted_prices, label="Predicted Prices", color="red", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Predicted Opening Price")
    plt.title(title)
    plt.legend()
    st.pyplot(plt)


# --- Streamlit UI ---
def main():
    st.title("📈 Stock Price Prediction with LSTM")

    # --- Model Initialization ---
    model = load_best_model(num_classes=2)

    # --- User Input for Start Date ---
    start_date = st.date_input("📅 Select Start Date for Prediction:", datetime.today())
    
    # --- Number of Years for Prediction ---
    num_years = st.number_input("🔮 Enter Number of Years to Predict:", min_value=1, max_value=10, value=5, step=1)
    num_days = num_years * 365

    # --- Generate Predictions ---
    if st.button("🚀 Predict Stock Prices"):
        st.write("🔍 Predicting next {} years...".format(num_years))

        # --- Dummy Input Data for Now (Random Data) ---
        input_data = np.random.rand(30, 18)  # Replace with real feature data

        # --- Make Predictions ---
        predictions = predict_future(model, input_data, num_days=num_days)

        # --- Create Future Dates ---
        future_dates = generate_future_dates(start_date, num_days)

        # --- Plot Predictions ---
        st.write("📊 Plotting Predicted Opening Prices...")
        plot_predictions(future_dates, predictions)

        # --- Download Predictions ---
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted_Open": predictions})
        csv_data = prediction_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
