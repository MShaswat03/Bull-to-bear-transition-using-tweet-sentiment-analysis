import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from model import initialize_model, save_model, load_model


# --- Save Results to CSV ---
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


# --- Model Training ---
def train_lstm_model(
    csv_file,
    feature_cols,
    label_col,
    sequence_length=30,
    batch_size=64,
    num_epochs=100,
    patience=20,
    learning_rate=0.001,
    model_save_path="results/best_model.pth"
):
    # --- Load Data ---
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)

    # --- Prepare Features & Labels ---
    feature_data = df[feature_cols].values
    labels = df[label_col].values
    company_data = df["company"].values
    date_data = df["date"].values
    open_data = df["open"].values
    close_data = df["close"].values

    # --- Create Sequences ---
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

    X, y, company_list, date_list, open_list, close_list = create_sequences(
        feature_data, labels, company_data, date_data, open_data, close_data, sequence_length
    )
    X = np.nan_to_num(X)

    # --- Split Data ---
    X_train, X_temp, y_train, y_temp, companies_train, companies_temp, dates_train, dates_temp, open_train, open_temp, close_train, close_temp = train_test_split(
        X, y, company_list, date_list, open_list, close_list, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, companies_val, companies_test, dates_val, dates_test, open_val, open_test, close_val, close_test = train_test_split(
        X_temp, y_temp, companies_temp, dates_temp, open_temp, close_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # --- Convert to PyTorch Tensors ---
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # ✅ --- Sanity Check ---
    print("✅ Checking train, val, and test splits...")
    print("Train class balance:", Counter(y_train.numpy()))
    print("Val class balance:", Counter(y_val.numpy()))
    print("Test class balance:", Counter(y_test.numpy()))
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # --- Load Data into Dataloader ---
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # --- Model Setup ---
    model = initialize_model(input_size=X_train.shape[2])
    criterion = nn.CrossEntropyLoss()  # This loss function expects class indices, not one-hot
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    best_val_f1 = 0
    early_stopping_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)  # Model should output logits
            loss = criterion(logits, y_batch)  # Ensure target labels are class indices (not one-hot)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).numpy()  # Take the class with highest score
            y_true_train.extend(y_batch.numpy())
            y_pred_train.extend(preds)

        # --- Train Metrics ---
        train_f1 = f1_score(y_true_train, y_pred_train, average="weighted")
        train_loss /= len(train_loader)

        # --- Validation ---
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

        # --- Save the Best Model ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(model, model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # --- Early Stopping ---
        if early_stopping_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}")
            break

    # --- Load the Best Model Before Testing ---
    load_model(model_save_path, model)

    # --- Evaluation After Early Stopping ---
    print("🎯 Evaluating model on test set...")
    model.eval()
    y_true_test, y_pred_test = [], []
    companies_test_pred, dates_test_pred, open_test_pred, close_test_pred = [], [], [], []

    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).numpy()
            y_true_test.extend(y_batch.numpy())
            y_pred_test.extend(preds)

            batch_size = X_batch.shape[0]
            companies_test_pred.extend(companies_test[i * batch_size: (i + 1) * batch_size])
            dates_test_pred.extend(dates_test[i * batch_size: (i + 1) * batch_size])
            open_test_pred.extend(open_test[i * batch_size: (i + 1) * batch_size])
            close_test_pred.extend(close_test[i * batch_size: (i + 1) * batch_size])

    # ✅ Save Results to CSV
    save_results_to_csv(
        y_true_test,
        y_pred_test,
        dates_test_pred,
        open_test_pred,
        close_test_pred,
        companies_test_pred
    )

    # ✅ Calculate Final Test F1 and Accuracy
    test_f1 = f1_score(y_true_test, y_pred_test, average="weighted")
    test_accuracy = accuracy_score(y_true_test, y_pred_test)

    # 🎯 Print Final Results
    print(f"🎯 Final Test Results - F1 Score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}")

    return model, test_f1, test_accuracy
