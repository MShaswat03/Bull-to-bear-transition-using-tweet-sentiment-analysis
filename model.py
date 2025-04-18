import torch
import torch.nn as nn

# ------------------------
# Attention Module
# ------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attn_output = torch.sum(attn_weights * lstm_output, dim=1)
        return attn_output

# ------------------------
# LSTM Classifier
# ------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        if use_attention:
            self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            out = self.attention(lstm_out)
        else:
            out = lstm_out[:, -1, :]
        return self.fc(out)

# ------------------------
# BiLSTM Classifier
# ------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        if use_attention:
            self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            out = self.attention(lstm_out)
        else:
            out = lstm_out[:, -1, :]
        return self.fc(out)

# ------------------------
# Seq2Seq Classifier
# ------------------------
class Seq2SeqClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(batch_size, 1, hidden.size(2)).to(x.device)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        return self.fc(decoder_output.squeeze(1))

# ------------------------
# Seq2Seq BiLSTM Classifier
# ------------------------
class Seq2SeqBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,
                               num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        _, (hidden, _) = self.encoder(x)
        forward = hidden[-2, :, :]
        backward = hidden[-1, :, :]
        hidden_cat = torch.cat((forward, backward), dim=1).unsqueeze(1)
        decoder_out, _ = self.decoder(hidden_cat)
        return self.fc(decoder_out.squeeze(1))

# ------------------------
# Model Initialization Dispatcher
# ------------------------
def initialize_model(input_size=18, hidden_size=256, num_layers=2, dropout=0.4, use_attention=True, model_type="lstm"):
    if model_type == "lstm":
        return LSTMClassifier(input_size, hidden_size, num_layers, num_classes=2, dropout=dropout, use_attention=use_attention)
    elif model_type == "bilstm":
        return BiLSTMClassifier(input_size, hidden_size, num_layers, num_classes=2, dropout=dropout, use_attention=use_attention)
    elif model_type == "seq2seq":
        return Seq2SeqClassifier(input_size=input_size)
    elif model_type == "seq2seq_bilstm":
        return Seq2SeqBiLSTMClassifier(input_size=input_size)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

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
