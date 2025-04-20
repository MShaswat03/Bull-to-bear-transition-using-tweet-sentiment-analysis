import matplotlib.pyplot as plt 
import json 

with open("results/model_scores.json") as f: 
    model_scores = json.load(f)

models = list(model_scores.keys())
f1_scores = [model_scores[m]["F1"] for m in models]
acc_scores  = [model_scores[m]["Accuracy"] for m in models]

plt.figure(figsize=(10, 6))
bar_width = 0.35
x = range(len(models))

plt.bar(x, f1_scores, width=bar_width, label="F1 Score")
plt.bar([i + bar_width for i in x], acc_scores, width=bar_width, label="Accuracy")

for i, v in enumerate(f1_scores): 
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', color='blue')
for i, v in enumerate(acc_scores):
    plt.text(i + bar_width, v + 0.005, f"{v:.4f}", ha='center', color='orange')

plt.xticks([i + bar_width / 2 for i in x], models)
plt.ylabel("Score")
plt.ylim(9.85, 1.0) 
plt.title("📊  Model Comparision: F1 Score vs Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.show()
