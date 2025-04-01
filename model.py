import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attn_output = torch.sum(attn_weights * lstm_output, dim=1)
        return attn_output


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, use_attention=True):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = Attention(hidden_size)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        if self.use_attention:
            out = self.attention(lstm_output)
        else:
            out = lstm_output[:, -1, :]
        out = self.fc(out)
        return out


def initialize_model(input_size=18, hidden_size=256, num_layers=3, dropout=0.4, use_attention=True):
    return LSTMClassifier(input_size, hidden_size, num_layers, num_classes=2, dropout=dropout, use_attention=use_attention)


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
