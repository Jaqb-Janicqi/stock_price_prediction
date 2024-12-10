import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

    def forward(self, x) -> torch.Tensor:
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)
        out, _ = self.gru(x, h_0)
        # Use only the output of the last time step
        out = self.fc(out[:, -1, :])
        return out
