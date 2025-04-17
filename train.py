import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from model_seqbi import initialize_model, save_model, load_model


def save_results_to_csv(y_true, y_pred, dates, open_prices, close_prices, companies):
    results_df = pd.DataFrame({
        "Date": dates,
        "Company": companies,
        "Open": open_prices,
        "Close": close_prices,
        "Actual": y_true,
        "Predicted": y_pred,
    })
    results_df.to_csv("results/predictions.csv", index=False)
    print("✅ Results saved to results/predictions.csv")


def train_lstm_model(
    csv_file,
    feature_cols,
    label_col,
    sequence_length=30,
    batch_size=64,
    num_epochs=50,
    patience=10,
    learning_rate=0.001,
    model_save_path="results/best_model.pth"
):
    # --- Load Data ---
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)

    feature_data = df[feature_cols].values
    labels = df[label_col].values
    company_data = df["company"].values
    date_data = df["date"].values
    open_data = df["open"].values
    close_data = df["close"].values

    def create_sequences(data, labels, companies, dates, open_prices, close_prices, seq_length):
        X, y, company_list, date_list, open_list, close_list = [], [], [], [], [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(labels[i + seq_length])
            company_list.append(companies[i + seq_length])
            date_list.append(dates[i + seq_length])
            open_list.append(open_prices[i + seq_length])
            close_list.append(close_prices[i + seq_length])
        return np.array(X), np.array(y), company_list, date_list, open_list, close_list

    # --- Create sequences ---
    X, y, company_list, date_list, open_list, close_list = create_sequences(
        feature_data, labels, company_data, date_data, open_data, close_data, sequence_length
    )
    X = np.nan_to_num(X)

    # --- Train/Val Split ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = initialize_model(input_size=X.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training ---
    best_val_f1 = 0
    early_stopping_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).numpy()
            y_true_train.extend(y_batch.numpy())
            y_pred_train.extend(preds)

        train_f1 = f1_score(y_true_train, y_pred_train, average="weighted")
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).numpy()
                y_true_val.extend(y_batch.numpy())
                y_pred_val.extend(preds)

        val_f1 = f1_score(y_true_val, y_pred_val, average="weighted")
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(model, model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}")
            break

    # --- Load Best Model ---
    load_model(model_save_path, model)

    # --- Full Inference ---
    print("🔍 Running full inference on all data...")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).numpy()
            all_preds.extend(preds)

    # --- Save Final Results ---
    save_results_to_csv(
        y_true=y,
        y_pred=all_preds,
        dates=date_list,
        open_prices=open_list,
        close_prices=close_list,
        companies=company_list
    )

    # --- Final metrics
    test_f1 = f1_score(y, all_preds, average="weighted")
    test_accuracy = accuracy_score(y, all_preds)
    print(f"🎯 Final Inference Results - F1 Score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}")

    return model, test_f1, test_accuracy
