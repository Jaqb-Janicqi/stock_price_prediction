import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._conv_stack = nn.Sequential(
            # Conv 1
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32,
                          kernel_size=(1, 2), stride=(1, 2)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32)
            ),
            # Conv 2
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32,
                          kernel_size=(1, 2), stride=(1, 2)),
                nn.Tanh(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.Tanh(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.Tanh(),
                nn.BatchNorm2d(32)
            ),
            # Conv 3
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32,
                          kernel_size=(1, 10)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
            )
        )
        self._inception_stacks = [
            # inception 1
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64,
                          kernel_size=(1, 1), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=(3, 1), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
            ),
            # inception 2
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64,
                          kernel_size=(1, 1), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=(5, 1), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
            ),
            # inception 3
            nn.Sequential(
                nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
                nn.Conv2d(in_channels=32, out_channels=64,
                          kernel_size=(1, 1), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
            )
        ]
        self._lstm = nn.LSTM(input_size=192, hidden_size=64,
                             num_layers=1, batch_first=True)
        self._linear = nn.Linear(64, 1)

    def forward(self, x) -> torch.Tensor:
        x = self._conv_stack(x)
        incept_out = [stack(x) for stack in self._inception_stacks]
        x = torch.stack(incept_out, dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self._lstm(x, (
            torch.zeros(1, x.size(0), 64, device=x.device),
            torch.zeros(1, x.size(0), 64, device=x.device)
        ))
        x = x[:, -1, :]
        return self._linear(x)
