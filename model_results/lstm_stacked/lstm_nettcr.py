import torch
import torch.nn as nn

class MultiSeedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_seeds, lstm_dropout):
        super().__init__()
        self.num_seeds = num_seeds
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_list = nn.ModuleList([
            nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout if num_layers > 1 else 0.0
            )
            for _ in range(num_seeds)
        ])
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        seed_outputs = []
        for lstm in self.lstm_list:
            _, (hn, _) = lstm(x)  # hn: (num_layers*2, batch_size, hidden_size)
            forward = hn[-2]      # last layer, forward
            backward = hn[-1]     # last layer, backward
            combined = torch.cat([forward, backward], dim=1)  # (batch_size, 2*hidden_size)
            seed_outputs.append(combined)
        
        return torch.cat(seed_outputs, dim=1)  # (batch_size, num_seeds * 2 * hidden_size)


class BindingPredictionBiModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_seeds,
        batch_size,
        num_sequences=7,
        lstm_dropout=0.3,
        ffn_dropout=0.3
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.hidden_size = hidden_size
        self.num_seeds = num_seeds

        # LSTM modules per sequence
        self.lstm_modules = nn.ModuleList([
            MultiSeedBiLSTM(input_size, hidden_size, num_layers, num_seeds, lstm_dropout)
            for _ in range(num_sequences)
        ])
        
        # Dropout after LSTM outputs (before FFN)
        self.post_lstm_dropout = nn.Dropout(ffn_dropout)

        ffn_input_dim = 2 * hidden_size * num_seeds * num_sequences
        self.classifier = nn.Sequential(
            nn.Linear(ffn_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(256, 1)  # Binary classification
        )
    
    def forward(self, x):
        # x: list of 7 tensors, each (batch_size, seq_len, input_size)
        lstm_outputs = []
        for i in range(self.num_sequences):
            out = self.lstm_modules[i](x[i])  # (batch_size, num_seeds * 2 * hidden_size)
            lstm_outputs.append(out)

        combined = torch.cat(lstm_outputs, dim=1)  # (batch_size, full_vector_dim)
        combined = self.post_lstm_dropout(combined)
        logit = self.classifier(combined)  # (batch_size, 1)
        return logit


input_size = 20
hidden_size = 64
num_layers = 2
num_seeds = 3
batch_size = 32
seq_len = 30
lstm_dropout = 0.2
ffn_dropout = 0.3

inputs = [torch.randn(batch_size, seq_len, input_size) for _ in range(7)]

model = BindingPredictionBiModel(
    input_size, hidden_size, num_layers, num_seeds, batch_size,
    lstm_dropout=lstm_dropout,
    ffn_dropout=ffn_dropout
)

output = model(inputs)
print(output.shape)  # (batch_size, 1)
