import torch
import torch.nn as nn

class Seq2SeqClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super(Seq2SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros((batch_size, 1, self.hidden_size), device=x.device)
        output, _ = self.decoder(decoder_input, (hidden, cell))
        output = self.fc(output.squeeze(1))
        return output

def initialize_model(input_size):
    return Seq2SeqClassifier(input_size=input_size, num_classes=2)


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
