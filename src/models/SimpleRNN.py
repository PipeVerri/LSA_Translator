import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=144, hidden_layers=7, output_dim=64):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # Para entrenamiento con batch y secuencias de diferentes longitudes

        packed = pack_padded_sequence(
            x, lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, h = self.rnn(packed)
        last = h[-1]
        logits = self.linear(last)
        return logits