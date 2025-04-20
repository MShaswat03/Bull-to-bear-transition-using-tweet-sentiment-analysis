import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

results_df = pd.read_csv("results/predictions.csv")
y_true = results_df["Actual"].values 
y_pred = results_df["Predicted"].values 

cm = confusion_matrix(y_true, y_pred) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negatve (0)", "Positive (1)"])
disp.plot(cmap=plt.cm.Blues) 
plt.title("Confusion Matrix for LSTM Modl")
plt.show()

os.makedirs("results", exist_ok=True)

plt.savefig("results/confusion_matrix_lstm.png", bbox_inches="tight")
plt.close()
