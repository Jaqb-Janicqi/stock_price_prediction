import torch
import torch.nn as nn


class lstm_tower_model(nn.Module):
    """
    Analysis of Bitcoin Price Prediction Using Machine Learning - Junwei Chen
    """

    def __init__(self, input_size, output_size) -> None:
        super(lstm_tower_model, self).__init__()
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
            nn.LSTM(256, 256, 1, batch_first=True),
            nn.Linear(256, output_size)
        ])

    def forward(self, x) -> torch.Tensor:
        for layer in self._lstm_tower:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        # Use only the output of the last time step
        return x[:, -1, :]
