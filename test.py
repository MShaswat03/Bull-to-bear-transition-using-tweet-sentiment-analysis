import torch
from model import initialize_model, load_model

# Initialize the model
model = initialize_model(input_size=18, hidden_size=256, num_layers=3, dropout=0.4, use_attention=True)

# Load the model
load_model("results/best_model.pth", model)

print("✅ Model loaded successfully!")

# Example of running a test (replace input_tensor with real input data)
# input_tensor shape: [batch_size, sequence_length, input_size]
# Example: torch.rand(1, 30, 18) -> batch_size=1, seq_length=30, input_size=18
input_tensor = torch.rand(1, 30, 18)  # Example input
output = model(input_tensor)
print(f"Predicted Output: {output}")
