import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=144,
            hidden_size=144,
            num_layers=5,
            batch_first=True
        )
        self.linear = nn.Linear(144, 64)

    def forward(self, x, lengths):
        # Para entrenamiento con batch y secuencias de diferentes longitudes
        lengths = lengths.cpu()

        packed = pack_padded_sequence(
            x, lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, h = self.rnn(packed)
        last = h[-1]
        logits = self.linear(last)
        return logits

    def forward_online(self, x, hidden=None):
        """
        Inferencia online: procesa frame a frame manteniendo el estado.

        Args:
            x: tensor de shape (batch, seq_len, input_size)
               Para un solo frame: (1, 1, 177)
            hidden: estado oculto previo de shape (num_layers, batch, hidden_size)
                    Si es None, se inicializa en ceros

        Returns:
            logits: predicción (batch, 64)
            hidden: nuevo estado oculto para la siguiente llamada
        """
        # Pasar el estado oculto anterior (o None para inicializar)
        out, hidden = self.rnn(x, hidden)

        # out: (batch, seq_len, hidden_size)
        # Tomar el último output de la secuencia
        last_out = out[:, -1, :]  # (batch, hidden_size)

        logits = self.linear(last_out)
        return logits, hidden