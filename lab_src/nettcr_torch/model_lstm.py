import torch
import torch.nn as nn

class MultiSeedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_seeds, lstm_dropout):
        super().__init__()
        self.lstm_list = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout if num_layers > 1 else 0.0
            )
            for _ in range(num_seeds)
        ])
    
    def forward(self, x):  # x: (batch, features, seq_len)
        x = x.transpose(1, 2)  # Convert to (batch, seq_len, features)
        seed_outputs = []
        for lstm in self.lstm_list:
            _, (hn, _) = lstm(x)  # hn: (2*num_layers, batch, hidden_size)
            forward = hn[-2]      # forward from last layer
            backward = hn[-1]     # backward from last layer
            combined = torch.cat([forward, backward], dim=1)  # (batch, 2*hidden_size)
            seed_outputs.append(combined)
        return torch.cat(seed_outputs, dim=1)  # (batch, num_seeds * 2 * hidden_size)


class BindingPredictionBiModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_seeds,
        batch_size,
        lstm_dropout=0.3,
        ffn_dropout=0.3
    ):
        super().__init__()
        self.batch_size = batch_size

        # Peptide input (first input): input_size = 20
        self.peptide_lstm = MultiSeedBiLSTM(
            input_size=20,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_seeds=num_seeds,
            lstm_dropout=lstm_dropout
        )

        # CDR inputs (6 inputs): input_size = 480
        self.cdr_lstm_modules = nn.ModuleList([
            MultiSeedBiLSTM(
                input_size=480,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_seeds=num_seeds,
                lstm_dropout=lstm_dropout
            )
            for _ in range(6)
        ])

        # Total input dim to FFN
        total_inputs = (1 + 6) * num_seeds * 2 * hidden_size
        self.post_lstm_dropout = nn.Dropout(ffn_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(total_inputs, 256),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(256, 1)  # Output logit
        )
    
    def forward(self, pep, a1, a2, a3, b1, b2, b3):
        x_list = [pep, a1, a2, a3, b1, b2, b3]
        # x_list[0] = peptide input: (batch, 20, seq_len)
        # x_list[1:] = 6 CDR inputs: each (batch, 480, seq_len)

        assert len(x_list) == 7, "Expected 7 input tensors (1 peptide + 6 CDRs)"

        outputs = []

        # Peptide
        peptide_out = self.peptide_lstm(x_list[0])  # (batch, num_seeds * 2 * hidden)
        outputs.append(peptide_out)

        # CDRs
        for i in range(6):
            cdr_out = self.cdr_lstm_modules[i](x_list[i + 1])  # (batch, num_seeds * 2 * hidden)
            outputs.append(cdr_out)

        combined = torch.cat(outputs, dim=1)  # (batch, total_features)
        combined = self.post_lstm_dropout(combined)
        logit = self.classifier(combined)  # (batch, 1)
        return logit.flatten()
