import torch
import torch.nn as nn

class Seq2SeqBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(Seq2SeqBiLSTMClassifier, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # ✅ BiLSTM encoder
        )

        # Decoder is unidirectional
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,  # input is the output of bidirectional encoder
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Encode input sequence with BiLSTM
        _, (hidden, cell) = self.encoder(x)  # hidden shape: [num_layers * 2, batch, hidden_size]

        # Merge forward and backward hidden states from last encoder layer
        # Last 2 layers: [-2] is forward, [-1] is backward
        forward_hidden = hidden[-2, :, :]   # shape: [batch, hidden_size]
        backward_hidden = hidden[-1, :, :]  # shape: [batch, hidden_size]
        hidden_cat = torch.cat((forward_hidden, backward_hidden), dim=1)  # shape: [batch, hidden_size * 2]

        # Prepare as input to decoder: shape [batch, 1, hidden_size * 2]
        decoder_input = hidden_cat.unsqueeze(1)

        # Decode one step
        decoder_output, _ = self.decoder(decoder_input)  # shape: [batch, 1, hidden_size]

        # Classify from decoder output
        logits = self.fc(decoder_output.squeeze(1))  # shape: [batch, num_classes]

        return logits

def initialize_model(input_size=18, hidden_size=128, num_layers=2, dropout=0.3):
    return Seq2SeqBiLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2
    )

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