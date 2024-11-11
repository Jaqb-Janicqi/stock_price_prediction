import torch
import torch.nn as nn


class LSTM_tower(nn.Module):
    """
    Analysis of Bitcoin Price Prediction Using Machine Learning - Junwei Chen
    """

    def __init__(self, input_size, output_size) -> None:
        super(LSTM_tower, self).__init__()
        self._lstm_tower = nn.ParameterList([
            nn.LSTM(input_size, 128, 1, batch_first=True),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.LSTM(128, 128, 1, batch_first=True),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.LSTM(128, 256, 1, batch_first=True),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.LSTM(256, 256, 1, batch_first=True)
        ])
        self._linear = nn.Linear(256, output_size)

    def forward(self, x) -> torch.Tensor:
        for layer in self._lstm_tower:
            if isinstance(layer, nn.LSTM):
                h_0 = torch.zeros(1, x.size(0), layer.hidden_size, device=x.device)
                c_0 = torch.zeros(1, x.size(0), layer.hidden_size, device=x.device)
                x, _ = layer(x, (h_0, c_0))
            else:
                x = layer(x)
        x = x[:, -1, :]
        return self._linear(x)
