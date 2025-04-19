import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ✅ Load predictions
results_df = pd.read_csv("results/predictions.csv")

# ✅ Extract actual and predicted values
y_true = results_df["Actual"].values
y_pred = results_df["Predicted"].values

# ✅ Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# ✅ Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative (0)", "Positive (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for BiLSTM Model")
plt.show()




