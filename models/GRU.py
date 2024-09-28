import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        out, _ = self.gru(x)
        # Use only the output of the last time step
        out = self.fc(out[:, -1, :])
        return out
