import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # Multiply by 2 for BiLSTM

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attn_output = torch.sum(attn_weights * lstm_output, dim=1)
        return attn_output

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, use_attention=True):
        super(BiLSTMClassifier, self).__init__()
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 🔁 Enable bidirectionality
        )

        if self.use_attention:
            self.attention = Attention(hidden_size)

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # x2 for BiLSTM output

    def forward(self, x):
        lstm_output, _ = self.lstm(x)  # Output shape: [batch, seq_len, hidden*2]

        if self.use_attention:
            out = self.attention(lstm_output)
        else:
            out = lstm_output[:, -1, :]  # Use last timestep (both directions)

        out = self.fc(out)
        return out

def initialize_model(input_size=18, hidden_size=128, num_layers=2, dropout=0.3, use_attention=True):
    return BiLSTMClassifier(input_size, hidden_size, num_layers, num_classes=2, dropout=dropout, use_attention=use_attention)

# ------------------------
# Save and Load Model
# ------------------------
def save_model(model, path="results/best_model.pth"):
    """Save the best model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")


def load_model(path, model):
    """Load model weights from the specified path."""
    model.load_state_dict(torch.load(path))
    print(f"✅ Model loaded from {path}")
    model.eval()
