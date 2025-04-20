import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassificer, GradientBoostingClassifier 
from sklearn.metrics import f1_score, accuracy_score 
from sklearn.model_selection import train_test_split 
import joblib 

def save_results_to_csv(y_true, y_pred, dates, open_prices, close_prices, companies, filename): 
    df = pd.DataFrame({ 
        "Date": dates, 
        "Company": companies, 
        "Open": open_prices, 
        "Actual": y_true,
        "Predicted": y_pred,
    })
    df.to_csv(filename, index=False)

def train_classic_model(csv_file, feature_cols, label_col, model_type="random_forest"): 
    df = pd.read_csv(csv_file) 
    df.dropna(inplace=True) 

    X = df[feature_cols] 
    y = df[label_col] 
    dates = df["date"]
    open_prices = df["open"] 
    close_prices = df["close"]
    companies = df["company"]

    X_train, X_test, y_train, y_test, dates_train, dates_test, open_train, open_test,  close_train, close_test, companies_train, companies_test = train_test_split(
        X, y, dates, open_prices, close_prices, companies, test_size=0.2, stratify=y, random_state=42
    )

    if model_type == "random_forest": 
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "gradient_boost":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    else: 
        raise ValueError("Unsupported model type")

    model.fit( X_train, y_train) 
    y_pred = model.predict(X_test) 

    acc = accuracy_score(y_test, y_pred) 
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f" {model_type.replace('_', ' '),title()} - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

    joblib.dump(model, f"results/{model_type}.pkl") 
    save_results_to_csv(y_test, y_pred, dates_test, open_test, close_test, companies_test, f"results/{model_type}_predictions.csv")

    return f1, acc


