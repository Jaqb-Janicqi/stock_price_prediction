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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

    def forward(self, x) -> torch.Tensor:
        h_0 = torch.zeros(self._lstm.num_layers, x.size(0), self._lstm.hidden_size, device=x.device)
        c_0 = torch.zeros(self._lstm.num_layers, x.size(0), self._lstm.hidden_size, device=x.device)
        out, _ = self._lstm(x, (h_0, c_0))
        out = self._fc(out[:, -1, :])
        return out
