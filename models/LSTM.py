import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        super(LSTM, self).__init__()
        self._lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self._fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x) -> torch.Tensor:
        out, _ = self._lstm(x)
        # Use only the output of the last time step
        out = self._fc(out[:, -1, :])
        return out
